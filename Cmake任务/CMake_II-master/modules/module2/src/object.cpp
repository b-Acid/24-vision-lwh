/**
 * @file argument.cpp
 * @author 赵曦 (535394140@qq.com)
 * @brief Object and object type based on OPC UA protocal
 * @version 1.0
 * @date 2022-09-17
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#include "robotlab/opcua_cs/ua_type/object.hpp"

using namespace std;
using namespace ua;

/**
 * @brief Construct a new Object Type object
 *
 * @param nameId Name of variable type nodeId
 */
ObjectType::ObjectType(const char *nameId) : __name(nameId)
{
    __nodeId = UA_NODEID_STRING(1, const_cast<char *>(nameId));
}

/**
 * @brief Construct a new Object Type object
 *
 * @param objType Object type of this object
 */
Object::Object(const ObjectType &objType) : __type(objType)
{
    const auto &name_variable = objType.getVariables();
    for (auto [name, variable] : name_variable)
        __name_variable[name] = variable;
}
