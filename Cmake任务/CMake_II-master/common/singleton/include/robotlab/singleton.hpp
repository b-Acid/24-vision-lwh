/**
 * @file singleton.hpp
 * @author 赵曦 (535394140@qq.com)
 * @brief 单例类模板
 * @version 1.0
 * @date 2022-02-08
 *
 * @copyright Copyright SCUT RobotLab(c) 2022
 *
 */

#pragma once

#include <assert.h>
#include <utility>

/**
 * @brief 单例类模板
 * 
 * @tparam T 需要变为单例的模板参数类型
 */
template <typename T>
class GlobalSingleton final
{
public:
    // instance
    static T *Get() { return *GetPPtr(); }

    // new
    template <typename... Args>
    static void New(Args &&...args)
    {
        assert(Get() == nullptr);
        *GetPPtr() = new T(std::forward<Args>(args)...);
    }

    // delete
    static void Delete()
    {
        if (Get() != nullptr)
        {
            delete Get();
            *GetPPtr() = nullptr;
        }
    }

private:
    static T **GetPPtr()
    {
        static T *ptr = nullptr;
        return &ptr;
    }
};