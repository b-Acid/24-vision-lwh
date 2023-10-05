/**
 * @file Math.cpp
 * @author 赵曦 (535394140@qq.com)
 * @brief 额外数据函数库
 * @version 1.0
 * @date 2021-06-14
 * 
 * @copyright Copyright SCUT RobotLab(c) 2021
 * 
 */

#include "Math.h"

using namespace std;
using namespace cv;

const float PI = 3.141592654f;
const float e = 2.7182818f;
const float g = 9.788f;

/**
 * @brief 获取角度差
 * 
 * @param angle_1 
 * @param angle_2 
 * @return 角度差 
 */
float getDeltaAngle(float angle_1, float angle_2)
{
    // 角度范围统一化
    while (fabs(angle_1) > 180.f)
    {
        angle_1 -= (angle_1 > 0.f) ? 360.f : -360.f;
    }
    while (fabs(angle_2) > 180.f)
    {
        angle_2 -= (angle_2 > 0.f) ? 360.f : -360.f;
    }
    // 计算差值
    float delta_angle = angle_1 - angle_2;
    if (angle_1 > 150.f && angle_2 < -150.f)
    {
        delta_angle -= 360.f;
    }
    else if (angle_1 < -150.f && angle_2 > 150.f)
    {
        delta_angle += 360.f;
    }
    return fabs(delta_angle);
}

/**
 * @note
 *                 【以下两个函数的公式推导】
 *         由针孔相机模型中的相似三角形关系推出下列公式:
 *    ((x,y)为图像坐标系下的坐标,(X,Y,Z)为相机坐标系下的坐标)
 *           x = fx × X/Z + Cx = fx ×  tan_yaw  + Cx
 *           y = fy × Y/Z + Cy = fy × tan_pitch + Cy
 *                              ||
 *                              ||  写成矩阵
 *                              || 相乘的方式
 *                              \/
 *   ┌ x ┐    1  ┌ fx 0  Cx ┐┌ X ┐   ┌ fx 0  Cx ┐   ┌ tan_yaw ┐
 *   │ y │ = ——— │ 0  fy Cy ││ Y │ = │ 0  fy Cy │ × │tan_pitch│
 *   └ 1 ┘    Z  └ 0  0  1  ┘└ Z ┘   └ 0  0  1  ┘   └    1    ┘
 *                              ||
 *                              ||
 *                              \/
 *           corMatrix = angelMatrix × cameraMatrix
 */

/**
 * @brief 用来获得相机中心相对于装甲板中心的角度
 * 
 * @param cameraMatrix 相机内参
 * @param distCoeff 畸变参数
 * @param center 图像中装甲板中心
 * @return x, y 方向夹角 -- 目标在图像右方，point.x 为正，目标在图像下方，point.y 为正
 */
Point2f calculateRelativeAngle(const Matx33f &cameraMatrix, const Matx51f &distCoeff, Point2f center)
{
    Matx31f tf_point;
    Matx33f cameraMatrix_inverse = cameraMatrix.inv();
    tf_point(0) = center.x;
    tf_point(1) = center.y;
    tf_point(2) = 1;
    // 得到tan角矩阵
    Matx31f tf_result = cameraMatrix_inverse * tf_point;
    // 从图像坐标系转换成世界坐标系角度
    return {rad2deg(atan(tf_result(0))),
            rad2deg(atan(tf_result(1)))};
}

/**
 * @brief 用来获得装甲板中心的像素坐标
 * 
 * @param cameraMatrix 相机内参
 * @param distCoeff 畸变参数
 * @param angle 目标与相机的夹角
 * @return x, y 坐标
 */
Point2f calculateRelativeCenter(const Matx33f &cameraMatrix, const Matx51f &distCoeff, Point2f angle)
{
    float yaw = tanf(deg2rad(angle.x));
    float pitch = tanf(deg2rad(angle.y));
    Matx31f center_vector;
    center_vector(0) = yaw;
    center_vector(1) = pitch;
    center_vector(2) = 1;
    //得到tan角矩阵
    Matx31f result = cameraMatrix * center_vector;
    return {result(0), result(1)};
}

/**
 * @brief 获取相机坐标系下小陀螺中心轴的位置
 *
 * @param cameraMatrix 相机内参
 * @param distCoeff 畸变参数
 * @param center 像素坐标系下的坐标
 * @param group 序列组
 * @return Point3f 
 */
Point3f calculateCameraCenter(const Matx33f &cameraMatrix, const Matx51f &distCoeff, const Point2f& center, float distance)
{
    // 转换至相机坐标系
    float angle_y = calculateRelativeAngle(cameraMatrix, distCoeff, center).y;
    // float angle_x = calculateRelativeAngle(cameraMatrix, distCoeff, center).x;
    Matx31f pixel_center_matrix (center.x, center.y, 1);
    Matx31f camera_center_matrix = cameraMatrix.inv() * distance * cosf(deg2rad(angle_y)) * pixel_center_matrix;
    // 输出
    Point3f camera_center;
    camera_center.x = camera_center_matrix(0);
    camera_center.y = camera_center_matrix(1);
    camera_center.z = camera_center_matrix(2);
    return camera_center;
}

