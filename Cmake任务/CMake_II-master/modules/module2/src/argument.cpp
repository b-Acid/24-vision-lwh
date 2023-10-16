/**
 * @file argument.cpp
 * @author 赵曦 (535394140@qq.com)
 * @brief
 * @version 1.0
 * @date 2022-09-17
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#include "robotlab/opcua_cs/ua_type/argument.hpp"

using namespace std;
using namespace ua;

/**
 * @brief Construct a new Argument object
 *
 * @param name Name of the argument
 * @param description Description of the argument
 * @param ua_types UA_TYPES_xxx
 * @param dimension_array Dimension array of variant
 */
Argument::Argument(const char *name, const char *description, int ua_types,
                   const vector<UA_UInt32> &dimension_array)
{
    UA_Argument_init(&__argument);
    __argument.description = UA_LOCALIZEDTEXT(const_cast<char *>("en-US"), const_cast<char *>(description));
    __argument.name = UA_STRING(const_cast<char *>(name));
    __argument.dataType = UA_TYPES[ua_types].typeId;
    __dim_arr = dimension_array;
    if (__dim_arr.empty())
        __argument.valueRank = UA_VALUERANK_SCALAR;
    else
    {
        __argument.valueRank = static_cast<UA_Int32>(__dim_arr.size());
        __argument.arrayDimensionsSize = __dim_arr.size();
        __argument.arrayDimensions = __dim_arr.data();
    }
}
