# 24-vision-lwh

## 1.装甲板识别
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
+ 封装了两个类如下
  
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
    float armored_plate_area;//装甲板面积下限，单位像素
    float armored_plate_length_width_ratio[2];//装甲板的长宽比阈值
    string filename;//文件名
public:
    detector() {};

    detector(int color,int threshold,float* lightdescriptor_length_width_ratio,float angle_diff,float length_diff,float armored_plate_area,float* armored_plate_length_width_ratio,string filename="null")
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
+ 使用dectecor类时，首先将其实例化，然后传入参数初始化detector类，直接调用detector.detect()开始识别，识别得到的文件将保存到同目录下的output.mp4。
