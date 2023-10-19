#  装甲板分类

### 分类原理

1. 用pytorch调用已经被广泛运用的resnet18，将数据集按照训练集：验证集=8：1进行训练。
2. 在RTX3060上训练15分钟，验证集上的准确率已达100%，这时训练集上准确率为97%。
3. 保存训练好的模型为ArmoClassificacion.pth,大小为134MB

### 改进

1. 数据集中的样本过于单调，导致了过拟合现象，模型在网上下载的随机图片上的识别表现并不好。
2. 模型还是比较大，resnet18中包含了两位数个卷积层，偏复杂。可以试试魔改resnet18，减少几个层。

### Classify.ipynb

+ 基于torch2.0.1+cuda11.7+torchvision0.15.2

#### 训练图像预处理：

+ 训练集随机旋转45°，增加亮度至150%，每个像素点以均值和方差均为0.5进行归一化。
+ 验证集只做每个像素点均值和方差均为0.5的归一化。
+ 注意，ToTensor()这步操作会自动把像素的rgb值都除以255。

```python
data_transfroms = { 
    'train':transforms.Compose([
        transforms.RandomRotation(45),##随机转45°
        #transforms.CenterCrop(),##中心剪裁
        ##transforms.RandomHorizontalFlip(p=0.5),##随机水平镜像
        ##transforms.RandomVerticalFlip(p=0.5),##随机垂直镜像
        ##transforms.ColorJitter(brightness=1.5,contrast=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ]),
    'valid':transforms.Compose([
        #transforms.Resize(256),##调整大小
        #transforms.CenterCrop(80),##中心剪裁
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

}
```

+ 训练集图片
+ ![训练集图片](https://github.com/b-Acid/Images/blob/main/train.png?raw=true)



#### 网络初始化

+ 本处调用torchvision从官网下载并搭建了了resnet18。由于本任务是6分类任务，将resnet18的最后一层变为512*6的全连接层，并添加一层softmax层来得到输出类别序号。

```python
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Sequential(
    (0): Linear(in_features=512, out_features=6, bias=True)
    (1): LogSoftmax(dim=1)
  )
)
```


#### 模型调用

+ 图像预处理
+ 神经网络接受3 * 80 * 80且已经归一化的矩阵，所以要对图像预处理，预处理函数如下：

```python
def process_image(image_path):
    img=Image.open(image_path)

    img=img.resize((80,80))

    img=np.array(img)/255

    mean=np.array([0.5,0.5,0.5])

    std=np.array([0.5,0.5,0.5])

    img=(img-mean)/std

    img=img.transpose((2,0,1))

    return img
```

+ 模型调用

```python
filename="3.png"
image1=process_image(filename)


output = model_ft(torch.unsqueeze(torch.FloatTensor(image1),0).to(device))
_,preds_tensor=torch.max(output,1)
preds=np.squeeze(preds_tensor.cpu().numpy())
flower_names[str(preds)]

print("Prediction:"+str(preds)+",True:"+filename[0])
imshow(image1,title="PRED:{},TRUE:{}".format(preds,filename[0]))
```

![识别效果图](https://github.com/b-Acid/Images/blob/main/output.png?raw=true)


#### [模型的onnxruntime部署见onnx部署仓库](https://github.com/b-Acid/24-vision-lwh/tree/main/onnx%E9%83%A8%E7%BD%B2)
