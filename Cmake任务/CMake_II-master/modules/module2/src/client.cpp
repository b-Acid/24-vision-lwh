/**
 * @file client.cpp
 * @author 赵曦 (535394140@qq.com)
 * @brief Industrial Internet server based on OPC UA protocol
 * @version 1.0
 * @date 2022-09-22
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#include <string>

#include "robotlab/opcua_cs/client.hpp"

using namespace std;
using namespace ua;

/**
 * @brief Construct a new Client object
 */
Client::Client()
{
    __client = UA_Client_new();
    UA_ClientConfig *config = UA_Client_getConfig(__client);
    UA_StatusCode status = UA_ClientConfig_setDefault(config);
    if (status != UA_STATUSCODE_GOOD)
        UA_Client_delete(__client);
}

/**
 * @brief Client connect to the specific server
 *
 * @param address Address of the server
 * @param username User name of the client
 * @param password Password of the client
 * @return Status code of this operation
 */
UA_Boolean Client::connect(const char *address, const char *username, const char *password)
{
    if (string(username).empty() || string(password).empty())
        return UA_Client_connect(__client, address) == UA_STATUSCODE_GOOD;
    return UA_Client_connectUsername(__client, address, username, password) == UA_STATUSCODE_GOOD;
}

/**
 * @brief Read specific variable from the server
 *
 * @param varName Variable name of the specific node
 * @return Specific Variable
 */
Variable Client::readVaiable(const char *varName)
{
    UA_NodeId nodeId = UA_NODEID_STRING(1, const_cast<char *>(varName));
    UA_Variant variant;
    UA_Variant_init(&variant);
    UA_StatusCode retval = UA_Client_readValueAttribute(__client, nodeId, &variant);
    if (retval != UA_STATUSCODE_GOOD)
        return Variable();
    // Variant information
    void *data = variant.data;
    UA_UInt32 ua_types = variant.type->typeKind;
    vector<UA_UInt32> dimension_array;
    UA_NodeId type_id = UA_NODEID_NUMERIC(0, UA_NS0ID_BASEDATAVARIABLETYPE);
    size_t array_size = variant.arrayDimensionsSize;
    if (array_size != 0)
    {
        dimension_array.reserve(array_size);
        for (size_t i = 0; i < array_size; ++i)
            dimension_array.emplace_back(variant.arrayDimensions[i]);
    }
    return Variable(data, ua_types, READ | WRITE, dimension_array, type_id);
}

/**
 * @brief Add variable data change notification callback to client
 *
 * @param varName Variable name of the specific node
 * @param dataChangeCallBack Data change callback function in client
 * @return Status code of this operation
 */
UA_Boolean Client::addVariableMonitor(const char *varName, UA_Client_DataChangeNotificationCallback dataChangeCallBack)
{
    UA_NodeId target = UA_NODEID_STRING(1, const_cast<char *>(varName));
    // Create a subscription
    UA_CreateSubscriptionRequest sub_request = UA_CreateSubscriptionRequest_default();
    UA_CreateSubscriptionResponse sub_response =
        UA_Client_Subscriptions_create(__client, sub_request, nullptr, nullptr, nullptr);
    UA_UInt32 subId = sub_response.subscriptionId;
    if (sub_response.responseHeader.serviceResult == UA_STATUSCODE_GOOD)
        UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_CLIENT, "Create subscription succeeded, id %u\n", subId);
    else
        return false;
    // Create a monitor
    UA_MonitoredItemCreateRequest mon_request = UA_MonitoredItemCreateRequest_default(target);
    UA_MonitoredItemCreateResult mon_response =
        UA_Client_MonitoredItems_createDataChange(__client, sub_response.subscriptionId, UA_TIMESTAMPSTORETURN_BOTH,
                                                  mon_request, &target, dataChangeCallBack, nullptr);
    if (mon_response.statusCode == UA_STATUSCODE_GOOD)
    {
        UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_CLIENT, "Monitoring '%s', id %u\n", varName,
                    mon_response.monitoredItemId);
    }
    else
        return false;

    return true;
}
