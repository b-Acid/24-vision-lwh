# 24-vision-lwh

##  1.装甲板识别
+ 基于OpenCV 4.7.0，通过轮廓检测，pnp等方法，实现了对装甲板的检测和姿态解算。

##  2.装甲板分类
+ 基于pytorch机器学习框架搭建了resnet18进行装甲板分类,使用软件的版本是torch2.0.1+cuda11.7+torchvision0.15.2。

## 3.SSH任务

+ 使用SSH进行两机间互传文件，编写configure文件使用别名实现快速登录。

## 4.CmakeⅠ任务

+ 首先在子文件夹编写CMakeLists.txt，将子目录里的程序编译成静态库，之后在主目录的CMakeLists.txt指定了所有头文件的路径和要链接的静态库，完成主函数的编译。

## 5.onnx部署模型任务

