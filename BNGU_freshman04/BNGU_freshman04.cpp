#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
using namespace cv;


/*
 * @function name: orderPoints
 * @作用：对四个点进行排序：左下，右下，右上，左上
 * @pts: 输入的点的集合（无序的点）
 * @return: 排列好的四个点的集合，如果不满足四个点的数量，返回原集合
 */
vector<Point2f> orderPoints(const vector<Point2f>& pts)
{
    vector<Point2f> rect(4);

    // 如果不满足点的数量，返回原集合
    if (pts.size() != 4)
    {
        return rect;
    }

    // 计算点的中心
    Point2f center(0, 0);
    for (auto pt : pts)
    {
        center.x += pt.x;
        center.y += pt.y;
    }
    center.x /= 4;
    center.y /= 4;

    // 根据点相对于中心的位置进行分类
    for (auto pt : pts)
    {
        // 左下
        if (pt.x < center.x && pt.y < center.y)
        {
            rect[0] = pt;
        }

        // 右下
        if (pt.x > center.x && pt.y < center.y)
        {
            rect[1] = pt;
        }

        // 右上
        if (pt.x > center.x && pt.y > center.y)
        {
            rect[2] = pt;
        }

        // 左上
        if (pt.x < center.x && pt.y > center.y)
        {
            rect[3] = pt;
        }


    }
    return rect;

}

int main()
{
    // 初始化摄像头
    VideoCapture capture(0);
    if (!capture.isOpened())
    {
        cerr << "无法打开摄像头" << endl;
        return -1;
    }

    // 设置相机内参
    Mat cameraMatrix = (Mat_<double>(3, 3) <<
        1000, 0, 640,
        0, 1000, 360,
        0, 0, 1);

    // 畸变参数，假设没有畸变
    Mat distCoeffs = Mat::zeros(4, 1, CV_32F);

    // 定义矩形的实际的物理尺寸（单位：米）
    float width = 0.05;
    float height = 0.025;

    // 定义矩形在世界坐标系终端四个角点坐标
    vector<Point3f> objectPoints =
    {
    {-width / 2, -height / 2, 0},// 左下
    {width / 2, -height / 2, 0}, // 右下
    {width / 2, height / 2, 0},  // 右上
    {-width / 2, height / 2, 0}, // 左上
    };

    Mat frame, gray, blurred, edges, img;
    while (true)
    {
        // 读取摄像头的帧
        capture >> frame;
        if (frame.empty())
        {
            cerr << "无法读取视频帧" << endl;
            return -1;
        }

        // 转化为灰度图像
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // 二值化处理
        Mat binary;
        threshold(gray, binary, 50, 255, THRESH_BINARY_INV | THRESH_OTSU);

        // 预处理
        GaussianBlur(binary, blurred, Size(5, 5), 0);
        Canny(blurred, edges, 50, 150);

        // 形态学操作，连接断开的边缘
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(edges, edges, MORPH_CLOSE, kernel);

        // 查找轮廓
        vector<vector<Point> > contours;
        findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        img = frame.clone();

        // 按面积排序，选取最大的五个轮廓
        sort(contours.begin(), contours.end(),
            [](const vector<Point>& a, const vector<Point>& b) {
                return contourArea(a) > contourArea(b);
            });
        if (contours.size() > 5) {
            contours.resize(5);
        }


        for (const auto& contour : contours)
        {
            // 计算轮廓周长
            double perimeter = arcLength(contour, true);

            // 过滤面积过小的轮廓
            if (perimeter < 100)
            {
                continue;
            }

            // 多边形拟合
            vector<Point> approx;
            approxPolyDP(contour, approx, 0.02 * perimeter, true);

            // 如果四边形是凸四边形
            if (approx.size() == 4 && isContourConvex(approx))
            {
                // 计算面积
                double area = contourArea(approx);

                // 过滤过小的面积
                if (area > 1000)
                {
                    // 转化成point2f格式
                    vector<Point2f> approxPoints;

                    for (auto& pts : approx)
                    {
                        approxPoints.emplace_back((float)pts.x, (float)pts.y);
                    }

                    // 对四个点进行排序
                    vector<Point2f> imagePoints = orderPoints(approxPoints);

                    // 绘制检测到的矩形
                    drawContours(img, vector<vector<Point>>{approx}, -1, Scalar(0, 255, 0), 3);

                    // 标记四个圆角点
                    for (auto& imagePoint : imagePoints)
                    {
                        circle(img, Point((int)imagePoint.x, (int)imagePoint.y), 5, Scalar(255, 0, 0), -1);
                    }

                    // 使用solvePnP估计姿态
                    Mat rvec, tvec;
                    bool success = solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec);

                    if (success)
                    {
                        // 定义坐标系的3D点
                        float axisLength = min(width, height) / 2;
                        vector<Point3f> axisPoints{
                            {0, 0, 0}, // 原点
                            {axisLength * 2, 0, 0}, // x轴
                            {0, -axisLength, 0},// y轴
                            {0, 0, -axisLength} // z轴
                        };

                        // 将3D的点投影到2D的平面图像上
                        vector<Point2f> project_points;
                        projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs, project_points);

                        // 提取投影后的点
                        Point origin = project_points[0];
                        Point x = project_points[1];
                        Point y = project_points[2];
                        Point z = project_points[3];

                        // 绘制坐标系
                        arrowedLine(img, origin, x, Scalar(0, 0, 255), 3);  // X轴 - 红色
                        arrowedLine(img, origin, y, Scalar(0, 255, 0), 3);  // y轴 - 绿色
                        arrowedLine(img, origin, z, Scalar(255, 0, 0), 3);  // z轴 - 蓝色

                        // 添加标签
                        putText(img, "X", x, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);  // X轴 - 红色
                        putText(img, "Y", y, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);  // y轴 - 绿色
                        putText(img, "Z", z, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2);  // z轴 - 蓝色


                        break;
                    }
                }
            }
        }
        // 显示边缘图像（用于调试）
        imshow("Edges", edges);
        imshow("Result", img);

        // 按'q'退出
        if (waitKey(1) == 'q') {
            break;
        }
    }
    // 释放内存
    capture.release();
    destroyAllWindows();
    return 0;
}