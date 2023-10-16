/**
 * @file ua_define.hpp
 * @author 赵曦 (535394140@qq.com)
 * @brief
 * @version 1.0
 * @date 2022-09-16
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#pragma once

#include "open62541.h"

namespace ua
{

// Access level redefine
enum AccessLevel : UA_Byte
{
    READ = UA_ACCESSLEVELMASK_READ,                     // UA_ACCESSLEVELMASK_READ
    WRITE = UA_ACCESSLEVELMASK_WRITE,                   // UA_ACCESSLEVELMASK_WRITE
    HISTORYREAD = UA_ACCESSLEVELMASK_HISTORYREAD,       // UA_ACCESSLEVELMASK_HISTORYREAD
    HISTORYWRITE = UA_ACCESSLEVELMASK_HISTORYWRITE,     // UA_ACCESSLEVELMASK_HISTORYWRITE
    SEMANTICCHANGE = UA_ACCESSLEVELMASK_SEMANTICCHANGE, // UA_ACCESSLEVELMASK_SEMANTICCHANGE
    STATUSWRITE = UA_ACCESSLEVELMASK_STATUSWRITE,       // UA_ACCESSLEVELMASK_STATUSWRITE
    TIMESTAMPWRITE = UA_ACCESSLEVELMASK_TIMESTAMPWRITE  // UA_ACCESSLEVELMASK_TIMESTAMPWRITE
};

// Type to UA_DataType
template <typename _Tp> inline const UA_DataType &getUaType() { return UA_TYPES[0]; }
// UA_Boolean to UA_DataType
template <> inline const UA_DataType &getUaType<UA_Boolean>() { return UA_TYPES[UA_TYPES_BOOLEAN]; }
// UA_SByte to UA_DataType
template <> inline const UA_DataType &getUaType<UA_SByte>() { return UA_TYPES[UA_TYPES_SBYTE]; }
// UA_Byte to UA_DataType
template <> inline const UA_DataType &getUaType<UA_Byte>() { return UA_TYPES[UA_TYPES_BYTE]; }
// UA_Int16 to UA_DataType
template <> inline const UA_DataType &getUaType<UA_Int16>() { return UA_TYPES[UA_TYPES_INT16]; }
// UA_UInt16 to UA_DataType
template <> inline const UA_DataType &getUaType<UA_UInt16>() { return UA_TYPES[UA_TYPES_UINT16]; }
// UA_Int32 to UA_DataType
template <> inline const UA_DataType &getUaType<UA_Int32>() { return UA_TYPES[UA_TYPES_INT32]; }
// UA_UInt32 to UA_DataType
template <> inline const UA_DataType &getUaType<UA_UInt32>() { return UA_TYPES[UA_TYPES_UINT32]; }
// UA_UInt64 to UA_DataType
template <> inline const UA_DataType &getUaType<UA_UInt64>() { return UA_TYPES[UA_TYPES_UINT64]; }
// UA_Float to UA_DataType
template <> inline const UA_DataType &getUaType<UA_Float>() { return UA_TYPES[UA_TYPES_FLOAT]; }
// UA_Double to UA_DataType
template <> inline const UA_DataType &getUaType<UA_Double>() { return UA_TYPES[UA_TYPES_DOUBLE]; }
// UA_String to UA_DataType
template <> inline const UA_DataType &getUaType<UA_String>() { return UA_TYPES[UA_TYPES_STRING]; }
// UA_DateTime to UA_DataType
template <> inline const UA_DataType &getUaType<UA_DateTime>() { return UA_TYPES[UA_TYPES_DATETIME]; }
// UA_Guid to UA_DataType
template <> inline const UA_DataType &getUaType<UA_Guid>() { return UA_TYPES[UA_TYPES_GUID]; }
// UA_Variant to UA_DataType
template <> inline const UA_DataType &getUaType<UA_Variant>() { return UA_TYPES[UA_TYPES_VARIANT]; }

} // namespace ua