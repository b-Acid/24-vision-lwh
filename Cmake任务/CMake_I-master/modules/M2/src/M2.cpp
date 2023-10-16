/**
 * @file M2.cpp
 * @author 赵曦 (535394140@qq.com)
 * @brief
 * @version 1.0
 * @date 2022-06-26
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#include "M2.h"
#include <iostream>

using namespace std;

M2::M2(float processErr, float measureErr)
{
    __filter = new KalmanFilter44(processErr, measureErr);
    cout << "M2: ";
    __a1.print2();
    __a2.push(0);
    cout << "size = " << static_cast<int>(__a2.size()) << endl;
}

M2::~M2()
{
    delete __filter;
    __filter = nullptr;
}