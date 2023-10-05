/**
 * @file Math.h
 * @author 赵曦 (535394140@qq.com)
 * @brief 额外数据函数库
 * @version 1.0
 * @date 2021-06-14
 *
 * @copyright Copyright SCUT RobotLab(c) 2021
 *
 */

#pragma once

#include <cmath>
#include <opencv2/core.hpp>

// --------------------【结构、类型、常量定义】--------------------

extern const float PI; // 圆周率
extern const float e;  // 自然对数底数
extern const float g;  // 重力加速度
// pnp 姿态解算
struct ResultPnP
{
    double yaw = 0;
    double pitch = 0;
    double roll = 0;
    cv::Mat tran_vec;
    cv::Mat rotat_vec;
    float distance = 0;
};

// 陀螺仪数据
struct GyroData
{
    float pitch = 0;       // 俯仰角度
    float yaw = 0;         // 偏转角度
    float pitch_speed = 0; // 俯仰角速度
    float yaw_speed = 0;   // 偏转角速度
};

// 定义 Matx11
namespace cv
{
    using Matx11f = Matx<float, 1, 1>;
    using Matx11d = Matx<double, 1, 1>;
    using Matx51f = Matx<float, 5, 1>;
    using Matx15f = Matx<float, 1, 5>;
}

// ------------------------【广义位移计算】------------------------
/**
 * @brief 获取距离
 *
 * @param pt_1 起始点
 * @param pt_2 终止点
 * @return float
 */
inline float getDistances(const cv::Point2f &pt_1, const cv::Point2f &pt_2)
{
    return sqrt(pow(pt_1.x - pt_2.x, 2) + pow(pt_1.y - pt_2.y, 2));
}

/**
 * @brief 获取与水平方向的夹角，以平面直角坐标系 x 轴为分界线，
 *        逆时针为正方向，范围: [-90°, 90°)，默认返回弧度制
 *
 * @param start 起点
 * @param end 终点
 * @param radian 返回弧度(默认)或角度
 * @return
 */
inline float getHAngle(const cv::Point2f &start, const cv::Point2f &end, bool radian = true)
{
    float rad = -atanf((end.y - start.y) / (end.x - start.x));
    return radian ? rad : rad * 180.0 / PI;
}

/**
 * @brief 获取与垂直方向的夹角，以平面直角坐标系 y 轴为分界线，
 *        顺时针为正方向，范围: (-90°, 90°]，默认返回弧度制
 *
 * @param start 起点
 * @param end 终点
 * @return radian 返回弧度(默认)或角度
 */
inline float getVerticalAngle(const cv::Point2f &start, const cv::Point2f &end, bool radian = true)
{
    float horizon = getHAngle(start, end);
    horizon = horizon < 0 ? -PI / 2.f - horizon : PI / 2.f - horizon;
    return radian ? horizon : horizon * 180.0 / PI;
}

float getDeltaAngle(float, float);                                                                       // 获取角度差
cv::Point2f calculateRelativeAngle(const cv::Matx33f &, const cv::Matx51f &, cv::Point2f);               // 计算相机中心相对于装甲板中心的角度
cv::Point2f calculateRelativeCenter(const cv::Matx33f &, const cv::Matx51f &, cv::Point2f);              // 计算装甲板中心的像素坐标
cv::Point3f calculateCameraCenter(const cv::Matx33f &, const cv::Matx51f &, const cv::Point2f &, float); // 已知像素坐标，计算装甲板中心的相机坐标

// ------------------------【常用数学公式】------------------------
// y = sec(x)
inline float sec(float x) { return 1.f / cosf(x); }
// y = csc(x)
inline float csc(float x) { return 1.f / sinf(x); }
// y = cot(x)
inline float cot(float x) { return 1.f / tanf(x); }
// ° -> rad
inline float deg2rad(float deg) { return deg * PI / 180.f; }
// rad -> °
inline float rad2deg(float rad) { return rad * 180.f / PI; }
// sgn 符号函数
inline int sgn(float x) { return (x > 0) ? 1 : ((x < 0) ? -1 : 0); }
// sigmoid函数
inline float sigmoid(float x, float k = 1, float A = 1, float miu = 0) { return A * (1 / (1 + powf(e, -k * x + miu))); }

// ------------------------【常用变换公式】------------------------
/**
 * @brief Matx 类型向量转化为 vector 类型向量
 *
 * @tparam _Mat Matx 类型
 * @tparam _Vec vector 类型
 * @param mat 修改前的向量
 * @param vec 修改后的向量
 */
template <typename _Mat, typename _Vec>
void matx2vec(const _Mat &mat, _Vec &vec)
{
    vec.clear();
    vec.resize(mat.cols * mat.rows);
    for (int i = 0; i < vec.size(); i++)
        vec[i] = mat(i);
}
