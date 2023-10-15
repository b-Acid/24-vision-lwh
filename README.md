# 24-vision-lwh

##  1.[装甲板识别](https://github.com/b-Acid/24-vision-lwh/tree/main/%E8%A3%85%E7%94%B2%E6%9D%BF%E8%AF%86%E5%88%AB)
+ 基于OpenCV 4.7.0，通过轮廓检测，pnp等方法，实现了对装甲板的检测和姿态解算。

##  2.[装甲板分类](https://github.com/b-Acid/24-vision-lwh/tree/main/%E8%A3%85%E7%94%B2%E6%9D%BF%E5%88%86%E7%B1%BB)
+ 基于pytorch机器学习框架搭建了resnet18进行装甲板分类,使用软件的版本是torch2.0.1+cuda11.7+torchvision0.15.2。

## 3.[SSH任务](https://github.com/b-Acid/24-vision-lwh/tree/main/SSH%E4%BB%BB%E5%8A%A1)

+ 使用SSH进行两机间互传文件，编写configure文件使用别名实现快速登录。

## 4.[CmakeⅠ任务](https://github.com/b-Acid/24-vision-lwh/tree/main/Cmake%E4%BB%BB%E5%8A%A1)

+ 首先在子文件夹编写CMakeLists.txt，将子目录里的程序编译成静态库，之后在主目录的CMakeLists.txt指定了所有头文件的路径和要链接的静态库，完成主函数的编译。

## 5.[onnx部署模型任务](https://github.com/b-Acid/24-vision-lwh/tree/main/onnx%E9%83%A8%E7%BD%B2)
+ 使用onnx-python将模型转化为onnx模式，并使用onnx-simplifier进行简化。用onnxruntime在c++中部署onnx模型。这里使用的是基于wideface数据集训练的只检测人脸的yolov5模型。实测对于720p视频，在3060负载40%的状态下，识别帧数超过100帧。
