/**
 * @file variable.hpp
 * @author 赵曦 (535394140@qq.com)
 * @brief Variable and variable type based on OPC UA protocal
 * @version 1.0
 * @date 2022-09-17
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#pragma once

#include <vector>

#include "ua_define.hpp"

namespace ua
{

// Variable based on OPC UA protocal
class Variable
{
    UA_Variant __variant;             // Variant
    std::vector<UA_UInt32> __dim_arr; // Dimension array of variant
    UA_Byte __access_level;           // Access level of this variable in server or client
    UA_NodeId __type_id;              // Variable type

public:
    Variable()
    {
        __type_id = UA_NODEID_NUMERIC(0, UA_NS0ID_BASEDATAVARIABLETYPE);
        UA_Variant_init(&__variant);
    }

    ~Variable() = default;

    Variable(const Variable &v)
        : __dim_arr(v.__dim_arr), __access_level(v.__access_level), __type_id(v.__type_id)
    {
        UA_Variant_copy(&v.__variant, &__variant);
    }

    Variable(Variable &&v) : __dim_arr(v.__dim_arr), __access_level(v.__access_level), __type_id(v.__type_id)
    {
        UA_Variant_copy(&v.__variant, &__variant);
    }

    void operator=(const Variable &v)
    {
        __dim_arr = v.__dim_arr;
        __type_id = v.__type_id;
        __access_level = v.__access_level;
        UA_Variant_copy(&v.__variant, &__variant);
    }

    /**
     * @brief Construct a new Variable object
     *
     * @tparam _InterType built-in type
     * @param val Built-in type data
     * @param access_level Accessibility of node data
     * @param type Variable type
     */
    template <typename _InterType>
    Variable(const _InterType &val, UA_Byte access_level = READ,
             const UA_NodeId &type = UA_NODEID_NUMERIC(0, UA_NS0ID_BASEDATAVARIABLETYPE))
        : __access_level(access_level), __type_id(type)
    {
        UA_Variant_init(&__variant);
        UA_Variant_setScalar(&__variant, const_cast<_InterType *>(&val), &getUaType<_InterType>());
    }

    /**
     * @brief Construct a new Variable object
     *
     * @param str string data
     * @param access_level Accessibility of node data
     * @param type_id Variable type
     */
    Variable(const char *str, UA_Byte access_level = READ,
             const UA_NodeId &type_id = UA_NODEID_NUMERIC(0, UA_NS0ID_BASEDATAVARIABLETYPE))
        : __access_level(access_level), __type_id(type_id)
    {
        UA_Variant_init(&__variant);
        UA_String ua_str = UA_STRING(const_cast<char *>(str));
        UA_Variant_setScalarCopy(&__variant, &ua_str, &UA_TYPES[UA_TYPES_STRING]);
    }

    /**
     * @brief Generating variable from raw data
     *
     * @param data Raw data
     * @param ua_types UA_TYPES_xxx
     * @param access_level Accessibility of node data
     * @param dimension_array Dimension array of variant
     * @param type NodeId of variable type
     */
    Variable(void *data, UA_UInt32 ua_types, UA_Byte access_level,
             const std::vector<UA_UInt32> &dimension_array = std::vector<UA_UInt32>(),
             const UA_NodeId &type = UA_NODEID_NUMERIC(0, UA_NS0ID_BASEDATAVARIABLETYPE));

    // Get nodeId of variable type
    inline const UA_NodeId &getNodeType() const { return __type_id; }
    // Get pointer of variant
    inline const UA_Variant &getVariant() const { return __variant; }
    // Get dimension
    inline std::size_t getDimension() const { return __dim_arr.size(); }
    // Get access level
    inline UA_Byte getAccess() const { return __access_level; }

private:
    /**
     * @brief Initialize the variable
     */
    void init();
};

// Variable type based on OPC UA protocal
class VariableType
{
    const char *__name;               // Name of variable type
    UA_NodeId __type_node;            // NodeId of the added variable
    UA_Variant __variant;             // The initial value
    std::vector<UA_UInt32> __dim_arr; // Dimension array of variant

public:
    VariableType();
    ~VariableType() = default;

    /**
     * @brief Construct a new Variable Type object
     *
     * @param vt Another instance of variable type
     */
    VariableType(const VariableType &vt);

    /**
     * @brief Construct a new Variable Type object
     *
     * @param vt Another instance of variable type
     */
    VariableType(VariableType &&vt);

    /**
     * @brief operator =
     *
     * @param vt Another instance of variable type
     */
    void operator=(const VariableType &vt);

    /**
     * @brief Construct a new Variable Type object
     *
     * @param nameId Name of variable type nodeId
     * @param variable Variable
     */
    VariableType(const char *nameId, const Variable &variable);

    // Get the name of the variable type
    inline const char *getName() const { return __name; }
    // Get the nodeId of the added variable
    inline UA_NodeId getNodeType() const { return __type_node; }
    // Get pointer of default variant
    UA_Variant getDefault() const { return __variant; }
    // Get dimension
    inline std::size_t getDimension() const { return __dim_arr.size(); }
};
} // namespace ua
