# ONNXRUNTIME部署yolov5实现人脸检测
### ONNXRUNTIME安装
### 模型的训练
1. 从github上克隆官方仓库，这里选择的是yolov5-6.0。
2. 下载数据集，这里选择的是[wider face数据集](http://shuoyang1213.me/WIDERFACE/)中的7900张图片。
   + 找到Download部分：
   ```
     ·WIDER Face Training Images [Google Drive] [Tencent Drive] [Hugging Face]
     ·WIDER Face Validation Images [Google Drive] [Tencent Drive] [Hugging Face]
     ·WIDER Face Testing Images [Google Drive] [Tencent Drive] [Hugging Face]
     ·Face annotations
   ```
   + 前三个文件夹是训练集，验证集，测试集，第四个文件是所有集的标注。
     
4. 整合数据包，划分为如下结构：
   ```
      datasets
      ├── images
      │   ├── train
      │   └── val
      └── labels
          ├── train
          └── val
   ```
   + 其中images下存储的是所有图片，labels下是所有图片的标注。
6. 配置.yaml文件
   + 在仓库目录的data文件夹中创建face.yaml，写入如下代码配置数据集路径和模型配置：
   ```yaml
    # Datasets
    train: datasets/images/train  # train images (relative to 'path')
    val: datasets/images/val  # val images (relative to 'path')
    test:  # test images (optional)
    
    # Classes
    nc: 1  # number of classes
    names: ['face']  # class names
   ```
7. 开始训练
    + 激活python核心，运行train.py。运行时指定yaml路径(--data参数)，训练epochs数(--epochs)，模型输入大小(--img)。
    ```
    source activate xxx (激活安装好了yolo需要的包的python核心)
    python train.py --data data/face.yaml --epochs 20 --img 320
    ```
    + 等待训练完成，训练好的模型和训练日志会被保存到run文件夹内。保存的模型为.pt格式，训练时的一些数据会以图表形式保存在相同目录下。
      

### 模型的转化
1. 在yolov5仓库中运行export.py，指定参数--weights(刚刚训练得到的权重文件)，--include onnx(保存为onnx文件)，--simplify(简化模型优化节点)。
  ```
  source activate xxx (激活安装好了yolo需要的包的python核心)
  python export.py --weights best.pt --include onnx --simplify
  ```
2. 得到的.onnx文件就可以调用onnxruntime的api在c++中部署了。
