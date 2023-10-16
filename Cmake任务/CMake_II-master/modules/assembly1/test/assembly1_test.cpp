/**
 * @file assembly1_test.cpp
 * @author 赵曦 (535394140@qq.com)
 * @brief
 * @version 1.0
 * @date 2022-08-29
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#include <iostream>

#include "robotlab/assembly1.h"

using namespace std;

int main()
{
    assembly1 ass(2);
    cout << ass.getNum() << endl;
    return 0;
}