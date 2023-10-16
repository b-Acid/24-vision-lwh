/**
 * @file client.cpp
 * @author 赵曦 (535394140@qq.com)
 * @brief
 * @version 1.0
 * @date 2022-10-01
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#include <iostream>
#include <thread>

#include "robotlab/opcua_cs.hpp"

using namespace std;
using namespace ua;

int main(int argc, char *argv[])
{
    Client client;
    if (!client.connect("opc.tcp://localhost:4840"))
        return 0;

    Variable dis = client.readVaiable("distance");
    float result = *reinterpret_cast<float *>(dis.getVariant().data);

    system("clear");
    cout << result << endl;
    this_thread::sleep_for(chrono::seconds(2));

    return 0;
}
