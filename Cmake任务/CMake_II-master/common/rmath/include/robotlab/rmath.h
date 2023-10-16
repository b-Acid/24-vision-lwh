/**
 * @file rmath.h
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
#include <unordered_map>

// --------------------【结构、类型、常量定义】--------------------
inline const float PI = 3.141592654f; // 圆周率
inline const float e = 2.7182818f;    // 自然对数底数
inline const float g = 9.788f;        // 重力加速度

// pnp 姿态解算
struct ResultPnP
{
    double yaw = 0;
    double pitch = 0;
    double roll = 0;
    cv::Vec3f t;
    cv::Matx33f R;
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
/**
 * @brief 正割
 *
 * @param x 自变量
 * @return sec(x)
 */
inline float sec(float x) { return 1.f / cosf(x); }

/**
 * @brief 余割
 *
 * @param x 自变量
 * @return csc(x)
 */
inline float csc(float x) { return 1.f / sinf(x); }

/**
 * @brief 余切
 *
 * @param x 自变量
 * @return cot(x)
 */
inline float cot(float x) { return 1.f / tanf(x); }

/**
 * @brief 角度转换为弧度
 *
 * @param deg 角度
 * @return 弧度
 */
inline float deg2rad(float deg) { return deg * PI / 180.f; }

/**
 * @brief 弧度转换为角度
 *
 * @param rad 弧度
 * @return 角度
 */
inline float rad2deg(float rad) { return rad * 180.f / PI; }

/**
 * @brief 符号函数
 *
 * @param x 自变量
 * @return 1    (x > 0)
 *         0    (x = 0)
 *         -1   (x < 0)
 */
template <typename _Tp>
inline _Tp sgn(_Tp x) { return (x > 0) ? 1 : ((x < 0) ? -1 : 0); }
// sigmoid函数
inline float sigmoid(float x, float k = 1, float A = 1, float miu = 0) { return A * (1 / (1 + powf(e, -k * x + miu))); }

/**
 * @brief 平面向量外积 (使用 Matx、Vec 模板)
 *
 * @tparam _Tp 向量类型
 * @param a 向量 A
 * @param b 向量 B
 * @return 外积
 */
template <typename _Tp>
typename _Tp::value_type cross2D(_Tp a, _Tp b) { return a(0) * b(1) - a(1) * b(0); }

/**
 * @brief 在指定范围内寻找众数，时间复杂度 O(N)
 *
 * @tparam _ForwardIterator 前向迭代器
 * @param _First 起始迭代器
 * @param _Last 终止迭代器
 * @return 众数
 */
template <typename _ForwardIterator>
typename _ForwardIterator::value_type
calculateModeNum(_ForwardIterator _First, _ForwardIterator _Last)
{
    using value_type = typename _ForwardIterator::value_type;
    std::unordered_map<value_type, size_t> hash_map;
    for (_ForwardIterator _it = _First; _it != _Last; ++_it)
        ++hash_map[*_it];
    return max_element(hash_map.begin(), hash_map.end(),
                       [&](const auto &pair_1, const auto &pair_2)
                       {
                           return pair_1.second < pair_2.second;
                       })
        ->first;
}

// ------------------------【常用变换公式】------------------------
