/**
 * @file assembly2.cpp
 * @author 赵曦 (535394140@qq.com)
 * @brief
 * @version 1.0
 * @date 2022-08-29
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#include "robotlab/assembly2.h"

using namespace std;
using namespace cv;

void assembly2::setP(Point2f p)
{
    __point = p;
}

Point2f assembly2::getP()
{
    return __point;
}
