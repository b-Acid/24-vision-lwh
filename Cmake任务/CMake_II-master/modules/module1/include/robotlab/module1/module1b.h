/**
 * @file module1b.h
 * @author 赵曦 (535394140@qq.com)
 * @brief
 * @version 1.0
 * @date 2022-08-29
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#pragma once

#include <memory>

#include "robotlab/assembly2.h"

class module1b
{
public:
    std::shared_ptr<assembly2> ass;

    module1b();
    void print();
};
