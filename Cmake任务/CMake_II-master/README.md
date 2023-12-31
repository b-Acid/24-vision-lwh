# CMake II 任务

给定一个代码框架，尝试创建缺少的 CMakeLists.txt 并编写，将此项目链接起来，最终可按照要求运行可执行文件

## 要求

#### CMake 部分变量命名要求

+ CMake 最小版本号：3.10

+ 项目名：Test

+ 可执行文件名：server（对应server.cpp）、client（对应client.cpp）

+ 库目标名：与文件夹名一致		例如：

  ```cmake
  add_library(ABC ${ABC_DIR})
  # 其中 ABC 文件夹如下
  # ABC
  # ├── include
  # │   └── ABC.h
  # └── src
  #     └── ABC.cpp
  ```

#### 注意事项

+ 禁止改动任何\*.h，\*.hpp，\*.cpp文件
+ CMake_Test_II/CMakeLists.txt 中的部分内容给出，禁止修改
+ 需要提前安装好OpenCV，一般情况下，其头文件的cmake参数集为`${OpenCV_INCLUDE_DIRS}`，动态库的cmake参数集为`${OpenCV_LIBS}`



## 运行

#### 程序编译步骤

```bash
mkdir build
cd build
cmake-gui ..
# 关闭 BUILD_A，打开 BUILD_B 和 BUILD_TESTS
make -j6
```



#### 测试

在当前终端运行 `ctest`，产生如下结果：

        Start 1: assembly1_test
    1/1 Test #1: assembly1_test ...................   Passed    0.00 sec
    
    100% tests passed, 0 tests failed out of 1
    
    Total Test time (real) =   0.00 sec


#### 程序运行步骤

```bash
./server
```

在当前路径下另开一个终端，运行：

```bash
./client
```

在 server 所在终端按 Ctrl + C 即可退出程序

#### 运行效果

+ ![](https://github.com/b-Acid/Images/blob/main/cmake2/2.jpg?raw=true)
+ ![](https://github.com/b-Acid/Images/blob/main/cmake2/1.jpg?raw=true)
