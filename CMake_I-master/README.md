# CMake任务

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

考核人员基本情况（防止有同学担心我们会故意刁难，其实不会）：

+ 考核用的电脑环境为Ubuntu20.04 LTS，并满足以下版本号要求：CMake 3.19，OpenCV 4.5.3

+ 考核人员会跳转至 CMake_Test 文件夹下，打开终端键入以下命令行，若不通过即视为失败

  ```bash
  mkdir build
  cd build
  cmake ..
  make -j6
  ./test
  ```

#### 最终参考运行效果

```
M1 construct
I'm M1
I'm A1
I'm A2
I'm A3
M2: I'm A2
size = 1
dis = 28.2843
M1 destruct
```
