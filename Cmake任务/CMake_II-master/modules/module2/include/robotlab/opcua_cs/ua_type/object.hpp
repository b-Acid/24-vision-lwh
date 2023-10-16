/**
 * @file argument.hpp
 * @author 赵曦 (535394140@qq.com)
 * @brief Object and object type based on OPC UA protocal
 * @version 1.0
 * @date 2022-09-17
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#pragma once

#include <unordered_map>

#include "variable.hpp"

namespace ua
{

// Variable and its name
using NameVarMap = std::unordered_map<const char *, Variable>;

// Object type based on OPC UA protocal
class ObjectType
{
    const char *__name;         // Name of object type
    UA_NodeId __nodeId;         // Actual nodeId of the added object
    NameVarMap __name_variable; // Hash table of name and variable

public:
    /**
     * @brief Construct a new Object Type object
     *
     * @param nameId Name of variable type nodeId
     */
    explicit ObjectType(const char *nameId);

    ~ObjectType() = default;

    /**
     * @brief Add variable to object type
     *
     * @param var_name The name of the variable
     * @param default_var The default variable that needs to be added to the object
     */
    inline void addDefault(const char *var_name, const Variable &default_var)
    {
        __name_variable[var_name] = default_var;
    }

    // Get the name of the nodeId
    inline const char *getName() const { return __name; }
    // Get the actual nodeId of the added object returned by Server after adding the object
    inline const UA_NodeId &getTypeNode() const { return __nodeId; }
    // Get hash table of name and variable from the object type
    inline const NameVarMap &getVariables() const { return __name_variable; }
};

// Object based on OPC UA protocal
class Object
{
    ObjectType __type;          // Object type
    NameVarMap __name_variable; // Hash table of name and variable

public:
    /**
     * @brief Construct a new Object Type object
     *
     * @param objType Object type of this object
     */
    explicit Object(const ObjectType &objType);

    ~Object() = default;

    /**
     * @brief Add variable to object type
     *
     * @param var_name The name of the variable
     * @param default_var The default variable that needs to be added to the object
     */
    inline void add(const char *var_name, const Variable &default_var)
    {
        if (__name_variable.find(var_name) != __name_variable.end())
            __name_variable[var_name] = default_var;
    }

    // Get variables and their names from the object type
    inline const NameVarMap &getVariables() const { return __name_variable; }
    // Get nodeId from this object type
    inline const UA_NodeId &getTypeNode() const { return __type.getTypeNode(); }
};

} // namespace ua