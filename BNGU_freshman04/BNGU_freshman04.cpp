#include <algorithm>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// 对四个点进行排序：左下，右下，右上，左上
vector<Point2f> orderPoints(const vector<Point2f>& pts) {
    vector<Point2f> rect(4);
    if (pts.size() != 4) return rect;

    // 计算点的中心
    Point2f center(0, 0);
    for (const auto& p : pts) {
        center.x += p.x;
        center.y += p.y;
    }
    center.x /= 4;
    center.y /= 4;

    // 根据点相对于中心的位置进行分类
    for (const auto& p : pts) {
        if (p.x < center.x && p.y > center.y) {
            rect[0] = p;  // 左下
        }
        else if (p.x > center.x && p.y > center.y) {
            rect[1] = p;  // 右下
        }
        else if (p.x > center.x && p.y < center.y) {
            rect[2] = p;  // 右上
        }
        else if (p.x < center.x && p.y < center.y) {
            rect[3] = p;  // 左上
        }
    }

    return rect;
}

int main() {
    // 初始化摄像头
    VideoCapture capture(0);
    if (!capture.isOpened()) {
        cerr << "无法打开摄像头" << endl;
        return -1;
    }

    

    // 设置相机内参
    Mat camera_matrix = (Mat_<float>(3, 3) <<
        1000, 0, 640,
        0, 1000, 360,
        0, 0, 1);

    // 畸变系数（假设无畸变）
    Mat dist_coeffs = Mat::zeros(4, 1, CV_32F);

    // 定义目标矩形的实际物理尺寸（单位：米）
    float width = 0.075;
    float height = 0.15;

    // 定义矩形在世界坐标系中的4个角点坐标（3D坐标）
    vector<Point3f> object_points = {
        {-width/2, -height/2, 0},// 左下
        {width / 2, -height / 2, 0},// 右下
        {width / 2, height / 2, 0},// 右上
        {-width / 2, height / 2, 0},// 左上
    };
       

    Mat frame, gray, blurred, edges, img;
    while (true) {
        // 读取摄像头帧
        capture >> frame;
        if (frame.empty()) {
            cerr << "无法获取帧" << endl;
            break;
        }

        // 转换为灰度图像
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        

        // 预处理
        GaussianBlur(gray, blurred, Size(5, 5), 0);
        Canny(blurred, edges, 50, 150);

        // 形态学操作，连接断开的边缘
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(edges, edges, MORPH_CLOSE, kernel);

        // 查找轮廓
        vector<vector<Point>> contours;
        findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        img = frame.clone();

        // 按面积排序，取最大的五个轮廓
        sort(contours.begin(), contours.end(),
            [](const vector<Point>& a, const vector<Point>& b) {
                return contourArea(a) > contourArea(b);
            });
        if (contours.size() > 5) {
            contours.resize(5);
        }

        // 初始化，标记是否找到目标矩形
        bool rectangle_found = false;

        for (const auto& contour : contours) {
            // 计算轮廓周长
            double perimeter = arcLength(contour, true);
            if (perimeter < 100) continue;  // 过滤过小的轮廓

            // 多边形拟合
            vector<Point> approx;
            approxPolyDP(contour, approx, 0.02 * perimeter, true);

            // 如果是四边形且是凸边形
            if (approx.size() == 4 && isContourConvex(approx)) {
                // 计算面积，过滤太小的区域
                double area = contourArea(approx);
                if (area > 1000) {
                    // 转换为Point2f类型
                    vector<Point2f> approx_points;
                    for (const auto& p : approx) {
                        approx_points.emplace_back((float)p.x, (float)p.y);
                    }

                    // 对4个点排序，确保与世界坐标系的角点顺序一致
                    vector<Point2f> image_points = orderPoints(approx_points);

                    // 绘制检测到的矩形
                    drawContours(img, vector<vector<Point>>{approx}, -1, Scalar(0, 255, 0), 3);

                    // 标记四个角点（蓝色圆点）并编号（0-3）
                    for (int i = 0; i < 4; ++i) {
                        circle(img, Point((int)image_points[i].x, (int)image_points[i].y),
                            5, Scalar(255, 0, 0), -1);
                        putText(img, to_string(i),
                            Point((int)image_points[i].x, (int)image_points[i].y),
                            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
                    }

                    // 使用solvePnP估计姿态
                    Mat rvec, tvec;
                    bool success = solvePnP(object_points, image_points,
                        camera_matrix, dist_coeffs, rvec, tvec);

                    if (success) {
                        // 定义坐标系的3D点（在矩形中心）
                        float axis_length = min(width, height) / 2;
                        vector<Point3f> axis_points{
                            {0, 0, 0},                 // 原点
                        {axis_length, 0, 0},       // X轴
                        {0, axis_length, 0},         // Y轴
                        {0, 0, axis_length},        // Z轴
                        };

                        // 将3D点投影到2D图像平面
                        vector<Point2f> projected_points;
                        projectPoints(axis_points, rvec, tvec,
                            camera_matrix, dist_coeffs, projected_points);

                        // 提取投影后的点
                        Point origin = projected_points[0];
                        Point x_axis = projected_points[1];
                        Point y_axis = projected_points[2];
                        Point z_axis = projected_points[3];

                        // 绘制坐标系
                        arrowedLine(img, origin, x_axis, Scalar(0, 0, 255), 3);  // X轴 - 红色
                        arrowedLine(img, origin, y_axis, Scalar(0, 255, 0), 3);  // Y轴 - 绿色
                        arrowedLine(img, origin, z_axis, Scalar(255, 0, 0), 3);  // Z轴 - 蓝色

                        // 添加标签
                        putText(img, "X", x_axis, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
                        putText(img, "Y", y_axis, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
                        putText(img, "Z", z_axis, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2);

                        rectangle_found = true;
                        break;  // 找到一个有效的矩形就跳出循环
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

    capture.release();
    cv::destroyAllWindows();
    return 0;
}