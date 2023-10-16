/**
 * @file variable.cpp
 * @author 赵曦 (535394140@qq.com)
 * @brief Variable and variable type based on OPC UA protocal
 * @version 1.0
 * @date 2022-09-17
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#include "robotlab/opcua_cs/ua_type/variable.hpp"

using namespace std;
using namespace ua;

/**
 * @brief Generating variable from raw data
 *
 * @param data Raw data
 * @param ua_types UA_TYPES_xxx
 * @param access_level Accessibility of node data
 * @param dimension_array Dimension array of variant
 * @param type_id NodeId of variable type
 */
Variable::Variable(void *data, UA_UInt32 ua_types, UA_Byte access_level, const vector<UA_UInt32> &dimension_array,
                   const UA_NodeId &type_id)
    : __access_level(access_level), __type_id(type_id)
{
    UA_Variant_init(&__variant);
    if (dimension_array.empty())
        UA_Variant_setScalar(&__variant, data, &UA_TYPES[ua_types]);
    else
    {
        size_t data_size = 1U;
        for (auto dim : dimension_array)
            data_size *= dim;
        UA_Variant_setArrayCopy(&__variant, data, data_size, &UA_TYPES[ua_types]);
    }
    __dim_arr = dimension_array;
    __variant.arrayDimensions = __dim_arr.data();
    __variant.arrayDimensionsSize = __dim_arr.size();
}

/**
 * @brief Construct a new Variable Type object
 */
VariableType::VariableType()
{
    __type_node = UA_NODEID_NUMERIC(0, UA_NS0ID_BASEDATAVARIABLETYPE);
    UA_Variant_init(&__variant);
}

/**
 * @brief Construct a new Variable Type object
 *
 * @param vt Another instance of variable type
 */
VariableType::VariableType(const VariableType &vt)
    : __name(vt.__name), __type_node(vt.__type_node), __dim_arr(vt.__dim_arr)
{
    UA_Variant_copy(&vt.__variant, &__variant);
}

/**
 * @brief Construct a new Variable Type object
 *
 * @param vt Another instance of variable type
 */
VariableType::VariableType(VariableType &&vt)
    : __name(vt.__name), __type_node(vt.__type_node), __dim_arr(vt.__dim_arr)
{
    UA_Variant_copy(&vt.__variant, &__variant);
}

/**
 * @brief operator =
 *
 * @param vt Another instance of variable type
 */
void VariableType::operator=(const VariableType &vt)
{
    __name = vt.__name;
    __dim_arr = vt.__dim_arr;
    __type_node = vt.__type_node;
    UA_Variant_copy(&vt.__variant, &__variant);
}

/**
 * @brief Construct a new Variable Type object
 *
 * @param nameId Name of variable type nodeId
 * @param variable Variable
 */
VariableType::VariableType(const char *nameId, const Variable &variable) : __name(nameId)
{
    __type_node = UA_NODEID_STRING(1, const_cast<char *>(nameId));

    UA_Variant_copy(&variable.getVariant(), &__variant);
    __dim_arr.reserve(__variant.arrayDimensionsSize);
    for (std::size_t i = 0; i < __variant.arrayDimensionsSize; ++i)
        __dim_arr.emplace_back(__variant.arrayDimensions[i]);
}
