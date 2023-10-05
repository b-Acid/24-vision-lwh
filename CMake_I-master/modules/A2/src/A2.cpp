/**
 * @file A2.cpp
 * @author 赵曦 (535394140@qq.com)
 * @brief
 * @version 1.0
 * @date 2022-06-26
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#include "A2.h"

/**
 * @brief push
 * 
 * @param num element
 */
void A2::push(int num)
{
    __vec.push_front(num);
}

/**
 * @brief pop
 * 
 * @return int 
 */
int A2::pop()
{
    int tmp = __vec.front();
    __vec.pop_back();
    return tmp;
}