/**
 * @file assembly2.h
 * @author 赵曦 (535394140@qq.com)
 * @brief
 * @version 1.0
 * @date 2022-08-29
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#pragma once

#include <opencv2/core.hpp>

class assembly2
{
    cv::Point2f __point;

public:
    void setP(cv::Point2f);
    cv::Point2f getP();
};
