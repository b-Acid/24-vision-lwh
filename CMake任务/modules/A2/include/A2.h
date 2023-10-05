/**
 * @file A2.h
 * @author 赵曦 (535394140@qq.com)
 * @brief
 * @version 1.0
 * @date 2022-06-26
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#pragma once

#include <deque>

class A2
{
    std::deque<int> __vec;

public:
    A2() = default;

    void push(int);
    int pop();

    inline size_t size() { return __vec.size(); }
};