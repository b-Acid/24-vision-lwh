# 24-vision-lwh

##  1.装甲板识别
### 识别原理
+ 1.对图像进行通道分离，分为BGR三个通道。
+ 2.按探测颜色对对应通道的图像进行二值化。
+ 3.对图像进行高斯滤波，减少噪点。
+ 4.用cv::dilate()进行膨胀，使轮廓边缘圆滑。
+ 5.使用漫水法使图像平滑。
+ 6.调用cv::findContours()进行轮廓检测。
+ 7.对每个轮廓做最小外接矩形，筛除不符合灯柱形状要求的矩形
+ 8.对所有轮廓从左到右匹配，每个轮廓和其右边的第一个轮廓匹配，按照预设参数匹配两个轮廓是否组成装甲板。
+ 9.标记匹配成功的装甲板，用pnp算法，cv::solvepnp()结算其相对相机的坐标，即可得到位姿
### dectect.cpp
+ 基于opencv4.7.0
+ 封装了两个类如下：
  
##### 灯管类
+ 属性：长，宽，面积，方向，中心坐标
```C++
class LightDescriptor{	    
public:float width, length, angle, area;
    Point2f center;
public:
    LightDescriptor() {};
    //给灯管套上一个旋转矩形
    LightDescriptor(const RotatedRect& light)
    {
        width = light.size.width;
        length = light.size.height;
        center = light.center;
        angle = light.angle;
        area = light.size.area();
    }
};
```

#### 探测器类
+ 属性：探测颜色，二值化阈值，灯柱长宽比上下限，角度差上限，长度偏差（百分比）上限，装甲板面积下限，装甲板长宽比上下限，传入文件的地址（可选,默认null，表示打开摄像头）
+ 功能：dectect()探测
```C++
class detector//探测器类
{
private:
   //探测器需要的参数主要是灯条颜色，亮度阈值，单灯条的长宽比，两灯条的角度差，装甲板的面积，装甲板的长宽比
    int color ;//0 red;1 green;2 blue;
    int threshold;//二值化阈值，0-255；
    float lightdescriptor_length_width_ratio[2];//单灯条的长宽比阈值
    float angle_diff;//角度差上限，单位度
    float length_diff;//两灯条长度偏差阈值
    float lightescriptor_area;//装甲板面积下限，单位像素
    float armored_plate_length_width_ratio[2];//装甲板的长宽比阈值
    string filename;//文件名
public:
    detector() {};

    detector(int color,int threshold,float* lightdescriptor_length_width_ratio,float angle_diff,float length_diff,float lightescriptor_area,float* armored_plate_length_width_ratio,string filename="null")
    {
        this->color=color;
        this->threshold=threshold;
        (this->lightdescriptor_length_width_ratio)[0]=lightdescriptor_length_width_ratio[0];
        (this->lightdescriptor_length_width_ratio)[1]=lightdescriptor_length_width_ratio[1];
        this->angle_diff=angle_diff;
        this->length_diff=length_diff;
        this->armored_plate_area=armored_plate_area;
        (this->armored_plate_length_width_ratio)[0]=armored_plate_length_width_ratio[0];
        (this->armored_plate_length_width_ratio)[1]=armored_plate_length_width_ratio[1];
        this->filename=filename;
    }

    void detect();
}
```
+ 使用dectecor类时，首先将其实例化，然后传入参数初始化detector类，直接调用detector.detect()开始识别，识别得到的文件将保存到同目录下的output.mp4。一个具体的例子如下：
+ 本例子中设置了探测器dec识别红色，阈值140，灯柱长宽比上下限分别为1和5，角度差最大值3°，长度差最多20%，灯柱最小面积20个像素，装甲板长宽比上下限分别为1和5。然后调用dec.detect()识别“testvideo.avi”中的装甲板

```C++
int main()
{
    int color =0;//识别红色
    int threshold=140;
    float lightdescriptor_length_width_ratio[2]={1,5};
    float angle_diff=3;
    float length_diff=0.2;
    float lightescriptor_area=20;
    float armored_plate_length_width_ratio[2]={1,4};


    detector dec(color,threshold,lightdescriptor_length_width_ratio,angle_diff,length_diff,lightescriptor_area,armored_plate_length_width_ratio,"testvideo.avi");
    dec.detect();
    return 0;
}
```


##  2.装甲板分类
### 分类原理
+ 1.用pytorch调用已经被广泛运用的resnet18，将数据集按照训练集：验证集=8：1进行训练。
+ 2.在RTX3060上训练15分钟，验证集上的准确率已达100%，这时训练集上准确率为97%。
+ 3.保存训练好的模型为ArmoClassificacion.pth,大小为134MB
### 改进
+ 1.数据集中的样本过于单调，导致了过拟合现象，模型在网上下载的随机图片上的识别表现并不好。
+ 2.模型还是比较大，resnet18中包含了两位数个卷积层，偏复杂。可以试试魔改resnet18，减少几个层。

### Classify.ipynb
+ 基于torch2.0.1+cuda11.7+torchvision0.15.2

#### 训练图像预处理：
+ 训练集随机旋转45°，增加亮度至150%，每个像素点以均值和方差均为0.5进行归一化。
+ 验证集只做每个像素点均值和方差均为0.5的归一化。
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



#### 网络初始化
+ 本处调用torchvision从官网下载并搭建了了resnet18。
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
+ 神经网络接受80*80且已经归一化的矩阵，所有要对图像预处理，预处理函数如下：
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
+模型调用

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

