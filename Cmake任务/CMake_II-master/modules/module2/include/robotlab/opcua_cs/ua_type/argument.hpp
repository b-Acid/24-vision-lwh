/**
 * @file argument.hpp
 * @author 赵曦 (535394140@qq.com)
 * @brief
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

// Method argument based on OPC UA protocol
class Argument
{
    UA_Argument __argument;           // Argument
    std::vector<UA_UInt32> __dim_arr; // Dimension array of variant

public:
    Argument(const Argument &arg) : __dim_arr(arg.__dim_arr) { UA_Argument_copy(&arg.__argument, &__argument); }
    Argument(Argument &&arg) : __dim_arr(arg.__dim_arr) { UA_Argument_copy(&arg.__argument, &__argument); }
    ~Argument() = default;

    /**
     * @brief Construct a new Argument object
     *
     * @param name Name of the argument
     * @param description Description of the argument
     * @param ua_types UA_TYPES_xxx
     * @param dimension_array Dimension array of variant
     */
    Argument(const char *name, const char *description, int ua_types,
             const std::vector<UA_UInt32> &dimension_array = std::vector<UA_UInt32>());

    // get argument
    UA_Argument getArgument() const { return __argument; }
    // get dimension
    inline std::size_t getDimension() const { return __dim_arr.size(); }
};

// Null argument
const std::vector<Argument> null_args;

} // namespace ua