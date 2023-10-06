/**
 * @file M1.cpp
 * @author 赵曦 (535394140@qq.com)
 * @brief
 * @version 1.0
 * @date 2022-06-26
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#include "M1.h"
#include <iostream>

using namespace std;

M1::M1()
{
    cout << "M1 construct" << endl;
}

M1::~M1()
{
    cout << "M1 destruct" << endl;
}

void M1::print()
{
    cout << "I'm M1" << endl;
    a.print1();
    a.print2();
    a.print3();
}