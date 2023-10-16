# CMake任务


### CMake 部分变量命名

+ CMake 最小版本号：3.10

+ 项目名：Test

+ 可执行文件名：test



#### 思路
+ 首先将common和modules里的cpp文件编译成两个静态库common和modules，然后在根目录里的CMakeLists.txt里链接这两个库。
+ 找到opencv的路径，链接opencv动态库。

#### 使用方法
+ 依次输入以下命令完成编译：
```
mkdir build
cd build
cmake ..
make -j8
./test

```

#### 最终运行效果

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
