/**
 * @file client.hpp
 * @author 赵曦 (535394140@qq.com)
 * @brief Industrial Internet client based on OPC UA protocol
 * @version 1.0
 * @date 2022-09-22
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#pragma once

#include "ua_type/argument.hpp"
#include "ua_type/object.hpp"
#include "ua_type/variable.hpp"

namespace ua
{

// Industrial Internet client based on OPC UA protocol
class Client final
{
    UA_Client *__client; // Client pointer

public:
    Client();

    /**
     * @brief Destroy the Client object
     */
    ~Client() { UA_Client_delete(__client); }

    /**
     * @brief UA_Client_run_iterate
     * 
     * @param timeOut Time out of the response of the server (ms)
     */
    inline void runIterate(UA_UInt32 timeOut) { UA_Client_run_iterate(__client, timeOut);}

    /**
     * @brief Client connect to the specific server
     *
     * @param address Address of the server
     * @param username User name of the client
     * @param password Password of the client
     * @return Status code of this operation
     */
    UA_Boolean connect(const char *address, const char *username = "", const char *password = "");

    /**
     * @brief Read specific variable from the server
     *
     * @param varName Variable name of the specific node
     * @return Specific Variable
     */
    Variable readVaiable(const char *varName);

    /**
     * @brief Add variable data change notification callback to client
     * 
     * @param varName Variable name of the specific node
     * @param dataChangeCallBack Data change callback function in client
     * @return Status code of this operation
     */
    UA_Boolean addVariableMonitor(const char *varName, UA_Client_DataChangeNotificationCallback dataChangeCallBack);
};

} // namespace ua
