/**
 * @file server.hpp
 * @author 赵曦 (535394140@qq.com)
 * @brief Industrial Internet server based on OPC UA protocol
 * @version 1.0
 * @date 2022-09-16
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

/**
 * @brief Industrial Internet server based on OPC UA protocol
 */
class Server final
{
    static void stopHandler(int sign);

    // Value callback, read function pointer definition
    using ValueCallBackRead = void (*)(UA_Server *, const UA_NodeId *, void *, const UA_NodeId *, void *,
                                       const UA_NumericRange *, const UA_DataValue *);
    // Value callback, write function pointer definition
    using ValueCallBackWrite = void (*)(UA_Server *, const UA_NodeId *, void *, const UA_NodeId *, void *,
                                        const UA_NumericRange *, const UA_DataValue *);
    // Data source callback, read function pointer definition
    using DataSourceRead = UA_StatusCode (*)(UA_Server *, const UA_NodeId *, void *, const UA_NodeId *, void *,
                                             UA_Boolean, const UA_NumericRange *, UA_DataValue *);
    // Data source callback, write function pointer definition
    using DataSourceWrite = UA_StatusCode (*)(UA_Server *, const UA_NodeId *, void *, const UA_NodeId *, void *,
                                              const UA_NumericRange *, const UA_DataValue *);

    UA_Server *__server = nullptr;                           // Server pointer
    static UA_Boolean __running;                             // Running status
    std::unordered_map<const char *, UA_NodeId *> __name_id; // Name:Id hash map

public:
    /**
     * @brief Construct a new Server object
     *
     * @param usr_passwd Vector of username:password
     */
    Server(const std::vector<UA_UsernamePasswordLogin> &usr_passwd = std::vector<UA_UsernamePasswordLogin>());
    ~Server();
    Server(const Server &) = delete;
    Server(Server &&) = delete;

    /**
     * @brief Run the server. This function is called last
     */
    void run();

    /**
     * @brief Add variable node to server
     *
     * @param varName Name of variable nodeId
     * @param data Variable data
     * @return Status code of this operation
     */
    UA_Boolean addVariableNode(const char *varName, const Variable &data);

    /**
     * @brief Add value call back function to variable node
     *
     * @param varName Name of variable nodeId
     * @param beforeRead Value callback, executed before reading
     * @param afterWrite Value callback, executed after writing
     * @return Status code of this operation
     */
    UA_Boolean addVariableNodeValueCallBack(const char *varName, ValueCallBackRead beforeRead,
                                            ValueCallBackWrite afterWrite);

    /**
     * @brief Add data source variable node to server
     *
     * @param varName Name of variable nodeId
     * @param data Variable data
     * @param onRead Data source callback, executed during reading
     * @param onWrite Data source callback, executed during writing
     * @return Status code of this operation
     */
    UA_Boolean addDataSourceVariableNode(const char *varName, const Variable &data, DataSourceRead onRead,
                                         DataSourceWrite onWrite);

    /**
     * @brief Add variable data change notification callback to server
     *
     * @param varName Name of variable nodeId
     * @param dataChangeCallBack Data change callback function in server
     * @param samplingInterval Sampling interval data
     * @return Status code of this operation
     */
    UA_Boolean addVariableMonitor(const char *varName, UA_Server_DataChangeNotificationCallback dataChangeCallBack,
                                  UA_Double samplingInterval);

    /**
     * @brief Add variable type node to server
     *
     * @param valType Variable type data
     * @return Status code of this operation
     */
    UA_Boolean addVariableTypeNode(const VariableType &valType);

    /**
     * @brief Add object node to server
     *
     * @param objName Name of object
     * @param data Object data
     * @return Status code of this operation
     */
    UA_Boolean addObjectNode(const char *objName, const Object &data);

    /**
     * @brief Add object type node to server
     *
     * @param objType Object type data
     * @return Status code of this operation
     */
    UA_Boolean addObjectTypeNode(const ObjectType &objType);

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
    UA_Boolean addMethodNode(const char *methodName, const char *description, UA_MethodCallback onMethod,
                             const std::vector<Argument> &inputArgs, const std::vector<Argument> &outputArgs);

private:
    /**
     * @brief Configure variable attribute before add variable node
     *
     * @param nameId Name of nodeId
     * @param data Variable data
     */
    UA_VariableAttributes configVariableAttribute(const char *nameId, const Variable &data);

    /**
     * @brief Get the child nodeId from the parent object nodeId
     *
     * @param parent_node Parent node of the object
     * @param target_name QualifiedName of the child nodeId
     * @return Child nodeId
     */
    UA_NodeId getChildNodeId(const UA_NodeId &parent_node, const UA_QualifiedName *target_name);
};

} // namespace ua
