/**
 * @file module1b.cpp
 * @author 赵曦 (535394140@qq.com)
 * @brief
 * @version 1.0
 * @date 2022-08-29
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#include <iostream>
#include <opencv2/core.hpp>

#include "robotlab/module1/module1b.h"

using namespace std;
using namespace cv;

module1b::module1b()
{
    ass = make_shared<assembly2>();
    ass->setP(Point2f(1.2f, 3.4f));
}

void module1b::print()
{
    cout << "ass = " << ass->getP() << endl;
}