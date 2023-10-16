/**
 * @file main.cpp
 * @author 赵曦 (535394140@qq.com)
 * @brief Industrial Internet server based on OPC UA protocol
 * @version 1.0
 * @date 2022-09-15
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#include "robotlab/opcua_cs/server.hpp"

using namespace std;
using namespace ua;

UA_Boolean Server::__running = false;

void Server::stopHandler(int sign)
{
    UA_LOG_INFO(UA_Log_Stdout, UA_LOGCATEGORY_USERLAND, "received ctrl-c");
    Server::__running = false;
}

/**
 * @brief Construct a new Server object
 *
 * @param usr_passwd Vector of username:password
 */
Server::Server(const std::vector<UA_UsernamePasswordLogin> &usr_passwd)
{
    if (__running)
        return;
    signal(SIGINT, stopHandler);
    signal(SIGTERM, stopHandler);

    __server = UA_Server_new();

    UA_ServerConfig *config = UA_Server_getConfig(__server);
    UA_ServerConfig_setDefault(config);

    if (!usr_passwd.empty())
    {
        config->accessControl.clear(&config->accessControl);
        UA_AccessControl_default(config, false, nullptr,
                                 &config->securityPolicies[config->securityPoliciesSize - 1].policyUri,
                                 usr_passwd.size(), usr_passwd.data());
    }
}

/**
 * @brief Destroy the Server:: Server object
 */
Server::~Server()
{
    for (auto [name, nodeId] : __name_id)
        UA_NodeId_delete(nodeId);
    UA_Server_delete(__server);
}

/**
 * @brief Run the server. This function is called last
 */
void Server::run()
{
    if (__running)
        return;
    __running = true;
    UA_Server_run(__server, &__running);
}

/**
 * @brief Configure variable attribute before add variable node
 *
 * @param varName Name of nodeId
 * @param data Variable data
 * @param accessLevel Accessibility of node data
 * @return UA_VariableAttributes
 */
UA_VariableAttributes Server::configVariableAttribute(const char *varName, const Variable &data)
{
    UA_VariableAttributes var_attr = UA_VariableAttributes_default;
    UA_Variant value = data.getVariant();
    UA_Variant_copy(&value, &var_attr.value);
    var_attr.dataType = value.type->typeId;
    if (data.getDimension() == 0)
        var_attr.valueRank = UA_VALUERANK_SCALAR;
    else
    {
        var_attr.arrayDimensions = value.arrayDimensions;
        var_attr.arrayDimensionsSize = value.arrayDimensionsSize;
        var_attr.valueRank = data.getDimension();
    }

    var_attr.description = UA_LOCALIZEDTEXT(const_cast<char *>("en-US"), const_cast<char *>(varName));
    var_attr.displayName = UA_LOCALIZEDTEXT(const_cast<char *>("en-US"), const_cast<char *>(varName));
    var_attr.accessLevel = data.getAccess();
    return var_attr;
}

/**
 * @brief Add variable node to server
 *
 * @param varName Name of variable nodeId
 * @param data Variable data
 * @return Status code of this operation
 */
UA_Boolean Server::addVariableNode(const char *varName, const Variable &data)
{
    if (__name_id.find(varName) != __name_id.end())
        return false;
    auto var_attr = configVariableAttribute(varName, data);
    // Add the variable node to the information model
    UA_NodeId *target_node_id = UA_NodeId_new();
    UA_StatusCode retval = UA_Server_addVariableNode(
        __server, UA_NODEID_STRING(1, const_cast<char *>(varName)), UA_NODEID_NUMERIC(0, UA_NS0ID_OBJECTSFOLDER),
        UA_NODEID_NUMERIC(0, UA_NS0ID_ORGANIZES), UA_QUALIFIEDNAME(1, const_cast<char *>(varName)), data.getNodeType(),
        var_attr, nullptr, target_node_id);
    __name_id[varName] = target_node_id;
    return retval == UA_STATUSCODE_GOOD;
}

/**
 * @brief Add value call back function to variable node
 *
 * @param nameId Name of variable nodeId
 * @param beforeRead Value callback, executed before reading
 * @param afterWrite Value callback, executed after writing
 * @return Status code of this operation
 */
UA_Boolean Server::addVariableNodeValueCallBack(const char *nameId, ValueCallBackRead beforeRead,
                                                ValueCallBackWrite afterWrite)
{
    // Can't find key then exit
    if (__name_id.find(nameId) == __name_id.end())
        return false;
    UA_ValueCallback callback;
    callback.onRead = beforeRead;
    callback.onWrite = afterWrite;
    UA_StatusCode retval = UA_Server_setVariableNode_valueCallback(__server, *__name_id[nameId], callback);
    return retval == UA_STATUSCODE_GOOD;
}

/**
 * @brief Add data source variable node to server
 *
 * @param varName Name of variable nodeId
 * @param data Variable data
 * @param onRead Data source callback, executed during reading
 * @param onWrite Data source callback, executed during writing
 * @return Status code of this operation
 */
UA_Boolean Server::addDataSourceVariableNode(const char *varName, const Variable &data, DataSourceRead onRead,
                                             DataSourceWrite onWrite)
{
    // Find key then exit
    if (__name_id.find(varName) != __name_id.end())
        return false;
    auto var_attr = configVariableAttribute(varName, data);
    // Add the variable node to the information model
    UA_DataSource data_source;
    data_source.read = onRead;
    data_source.write = onWrite;
    UA_NodeId *target_node_id = UA_NodeId_new();
    UA_StatusCode retval = UA_Server_addDataSourceVariableNode(
        __server, UA_NODEID_STRING(1, const_cast<char *>(varName)), UA_NODEID_NUMERIC(0, UA_NS0ID_OBJECTSFOLDER),
        UA_NODEID_NUMERIC(0, UA_NS0ID_ORGANIZES), UA_QUALIFIEDNAME(1, const_cast<char *>(varName)),
        UA_NODEID_NUMERIC(0, UA_NS0ID_BASEDATAVARIABLETYPE), var_attr, data_source, nullptr, nullptr);
    __name_id[varName] = target_node_id;
    return retval == UA_STATUSCODE_GOOD;
}

/**
 * @brief Add variable data change notification callback to server
 *
 * @param varName Name of variable nodeId
 * @param dataChangeCallBack Data change callback function in server
 * @param samplingInterval Sampling interval data
 * @return Status code of this operation
 */
UA_Boolean Server::addVariableMonitor(const char *varName, UA_Server_DataChangeNotificationCallback dataChangeCallBack,
                                      UA_Double samplingInterval)
{
    // Can't find key then exit
    if (__name_id.find(varName) == __name_id.end())
        return false;
    UA_NodeId *target_node_id = __name_id[varName];
    UA_MonitoredItemCreateRequest mon_request = UA_MonitoredItemCreateRequest_default(*target_node_id);
    // Sampling interval (ms)
    mon_request.requestedParameters.samplingInterval = samplingInterval;
    UA_MonitoredItemCreateResult mon_result = UA_Server_createDataChangeMonitoredItem(
        __server, UA_TIMESTAMPSTORETURN_BOTH, mon_request, target_node_id, dataChangeCallBack);
    return mon_result.statusCode == UA_STATUSCODE_GOOD;
}

/**
 * @brief Add variable type node to server
 *
 * @param dataType Variable type data
 * @return Status code of this operation
 */
UA_Boolean Server::addVariableTypeNode(const VariableType &dataType)
{
    const char *typeName = dataType.getName();
    if (__name_id.find(typeName) != __name_id.end())
        return false;
    UA_VariableTypeAttributes type_attr = UA_VariableTypeAttributes_default;
    UA_Variant value = dataType.getDefault();
    UA_Variant_copy(&value, &type_attr.value);
    type_attr.dataType = value.type->typeId;
    if (dataType.getDimension() == 0)
        type_attr.valueRank = UA_VALUERANK_SCALAR;
    else
    {
        type_attr.arrayDimensions = value.arrayDimensions;
        type_attr.arrayDimensionsSize = value.arrayDimensionsSize;
        type_attr.valueRank = dataType.getDimension();
    }
    type_attr.displayName = UA_LOCALIZEDTEXT(const_cast<char *>("en-US"), const_cast<char *>(typeName));
    UA_NodeId *target_node_id = UA_NodeId_new();
    UA_StatusCode retval = UA_Server_addVariableTypeNode(
        __server, dataType.getNodeType(), UA_NODEID_NUMERIC(0, UA_NS0ID_BASEDATAVARIABLETYPE),
        UA_NODEID_NUMERIC(0, UA_NS0ID_HASSUBTYPE), UA_QUALIFIEDNAME(1, const_cast<char *>(typeName)), UA_NODEID_NULL,
        type_attr, nullptr, target_node_id);
    __name_id[typeName] = target_node_id;
    return retval == UA_STATUSCODE_GOOD;
}

/**
 * @brief Add object node to server
 *
 * @param objName Name of object
 * @param data Object data
 * @return Status code of this operation
 */
UA_Boolean Server::addObjectNode(const char *objName, const Object &data)
{
    if (__name_id.find(objName) != __name_id.end())
        return false;
    UA_ObjectAttributes obj_attr = UA_ObjectAttributes_default;
    obj_attr.displayName = UA_LOCALIZEDTEXT(const_cast<char *>("en-US"), const_cast<char *>(objName));
    // Object NodeId
    UA_NodeId *obj_node_id = UA_NodeId_new();
    UA_StatusCode retval;
    retval = UA_Server_addObjectNode(__server, UA_NODEID_NULL, UA_NODEID_NUMERIC(0, UA_NS0ID_OBJECTSFOLDER),
                                     UA_NODEID_NUMERIC(0, UA_NS0ID_ORGANIZES),
                                     UA_QUALIFIEDNAME(1, const_cast<char *>(objName)), data.getTypeNode(), obj_attr,
                                     NULL, obj_node_id);
    if (retval != UA_STATUSCODE_GOOD)
        return false;
    __name_id[objName] = obj_node_id;
    // Get [name:variable]
    const auto &name_variables = data.getVariables();
    UA_Variant variant;
    for (const auto [name, variable] : name_variables)
    {
        // Get child NodeId
        UA_QualifiedName qualified_name = UA_QUALIFIEDNAME(1, const_cast<char *>(name));
        UA_NodeId child_id = getChildNodeId(*obj_node_id, &qualified_name);
        // Set variable
        UA_Variant_init(&variant);
        UA_Variant_copy(&variable.getVariant(), &variant);
        retval = UA_Server_writeValue(__server, child_id, variant);
        if (retval != UA_STATUSCODE_GOOD)
            return false;
    }
    return true;
}

/**
 * @brief Add object type node to server
 *
 * @param objType Object type data
 * @return Status code of this operation
 */
UA_Boolean Server::addObjectTypeNode(const ObjectType &objType)
{
    const char *objTypeName = objType.getName();
    // Define the object type node
    UA_ObjectTypeAttributes obj_attr = UA_ObjectTypeAttributes_default;
    obj_attr.displayName = UA_LOCALIZEDTEXT(const_cast<char *>("en-US"), const_cast<char *>(objTypeName));
    UA_NodeId *target_node_id = UA_NodeId_new();
    UA_StatusCode retval = UA_Server_addObjectTypeNode(
        __server, objType.getTypeNode(), UA_NODEID_NUMERIC(0, UA_NS0ID_BASEOBJECTTYPE),
        UA_NODEID_NUMERIC(0, UA_NS0ID_HASSUBTYPE), UA_QUALIFIEDNAME(1, const_cast<char *>(objTypeName)), obj_attr,
        nullptr, target_node_id);
    if (retval != UA_STATUSCODE_GOOD)
        return false;
    __name_id[objTypeName] = target_node_id;
    // Add variable node to object type as child node
    auto var_names = objType.getVariables();
    for (const auto [varName, variable] : var_names)
    {
        // Add variable attributes
        UA_VariableAttributes attr = UA_VariableAttributes_default;
        attr.displayName = UA_LOCALIZEDTEXT(const_cast<char *>("en-US"), const_cast<char *>(varName));
        attr.accessLevel = variable.getAccess();
        UA_Variant value = variable.getVariant();
        UA_Variant_copy(&value, &attr.value);
        UA_NodeId nodeId;
        retval = UA_Server_addVariableNode(__server, UA_NODEID_NULL, objType.getTypeNode(),
                                           UA_NODEID_NUMERIC(0, UA_NS0ID_HASCOMPONENT),
                                           UA_QUALIFIEDNAME(1, const_cast<char *>(varName)),
                                           UA_NODEID_NUMERIC(0, UA_NS0ID_BASEDATAVARIABLETYPE), attr, nullptr, &nodeId);
        if (retval != UA_STATUSCODE_GOOD)
            return false;
        /* Make the student name mandatory */
        retval = UA_Server_addReference(__server, nodeId, UA_NODEID_NUMERIC(0, UA_NS0ID_HASMODELLINGRULE),
                                        UA_EXPANDEDNODEID_NUMERIC(0, UA_NS0ID_MODELLINGRULE_MANDATORY), true);
        if (retval != UA_STATUSCODE_GOOD)
            return false;
    }
    return true;
}

/**
 * @brief Add method node to server
 *
 * @param methodName Name of method nodeId
 * @param description Description of the argument
 * @param onMethod Function body of method call back
 * @param inputArgs Vector of input arguments
 * @param outputArgs Vector of output arguments
 * @return Status code of this operation
 */
UA_Boolean Server::addMethodNode(const char *methodName, const char *description, UA_MethodCallback onMethod,
                                 const vector<Argument> &inputArgs, const vector<Argument> &outputArgs)
{
    if (__name_id.find(methodName) != __name_id.end())
        return false;
    UA_MethodAttributes method_attr = UA_MethodAttributes_default;
    method_attr.description = UA_LOCALIZEDTEXT(const_cast<char *>("en-US"), const_cast<char *>(description));
    method_attr.displayName = UA_LOCALIZEDTEXT(const_cast<char *>("en-US"), const_cast<char *>(methodName));
    method_attr.executable = true;
    method_attr.userExecutable = true;
    // Extract core data: UA_Argument
    vector<UA_Argument> inputs;
    inputs.reserve(inputArgs.size());
    for (auto &inputArg : inputArgs)
        inputs.emplace_back(inputArg.getArgument());
    vector<UA_Argument> outputs;
    outputs.reserve(outputArgs.size());
    for (auto &outputArg : outputArgs)
        outputs.emplace_back(outputArg.getArgument());
    // Add method node
    UA_NodeId *target_node_id = UA_NodeId_new();
    UA_StatusCode retval = UA_Server_addMethodNode(
        __server, UA_NODEID_STRING(1, const_cast<char *>(methodName)), UA_NODEID_NUMERIC(0, UA_NS0ID_OBJECTSFOLDER),
        UA_NODEID_NUMERIC(0, UA_NS0ID_HASCOMPONENT), UA_QUALIFIEDNAME(1, const_cast<char *>(methodName)), method_attr,
        onMethod, inputArgs.size(), inputs.data(), outputArgs.size(), outputs.data(), nullptr, target_node_id);
    __name_id[methodName] = target_node_id;
    return retval == UA_STATUSCODE_GOOD;
}

/**
 * @brief Get the child nodeId from the parent object nodeId
 *
 * @param parent_node Parent node of the object
 * @param target_name QualifiedName of the child nodeId
 * @return Child nodeId
 */
UA_NodeId Server::getChildNodeId(const UA_NodeId &parent_node, const UA_QualifiedName *target_name)
{
    auto browse_path = UA_Server_browseSimplifiedBrowsePath(__server, parent_node, 1, target_name);
    UA_NodeId ret_node;
    UA_NodeId_copy(&browse_path.targets->targetId.nodeId, &ret_node);
    return ret_node;
}
