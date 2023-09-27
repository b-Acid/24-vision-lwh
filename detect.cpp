#include "stdio.h"
#include<iostream> 
#include <cmath> 
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<string>
using namespace std;
using namespace cv;


class LightDescriptor//灯管类
{	    
public:float width, length, angle, area;
    Point2f center;
public:
    LightDescriptor() {};
    //让得到的灯条套上一个旋转矩形，以方便之后对角度这个特殊因素作为匹配标准
    LightDescriptor(const RotatedRect& light)
    {
        width = light.size.width;
        length = light.size.height;
        center = light.center;
        angle = light.angle;
        area = light.size.area();
    }
};


class detector//探测器类
{
private:
   //探测器需要的参数主要是灯条颜色，亮度阈值，单灯条的长宽比，两灯条的角度差，装甲板的面积，装甲板的长宽比
    int color ;//0 red;1 green;2 blue;
    int threshold;//二值化阈值，0-255；
    float lightdescriptor_length_width_ratio[2];//单灯条的长宽比阈值
    float angle_diff;//角度差下限，单位度
    float length_diff;//两灯条长度偏差阈值
    float armored_plate_area;//装甲板面积下限，单位像素
    float armored_plate_length_width_ratio[2];//装甲板的长宽比阈值


    string filename="null";//文件名


public:
    detector() {};
    detector(int color,int threshold,float* lightdescriptor_length_width_ratio,float angle_diff,float length_diff,float armored_plate_area,float* armored_plate_length_width_ratio)
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
    }

    detector(int color,int threshold,float* lightdescriptor_length_width_ratio,float angle_diff,float length_diff,float armored_plate_area,float* armored_plate_length_width_ratio,string filename)
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

    void detect()
    {
        VideoCapture video;
        if(filename=="null"){
            video.open(0);//默认开启摄像头
        }
        else{
            video.open(filename);
        }




        Mat frame, channels[3], binary, Gaussian, dilatee;
        Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        

        
        VideoWriter writer;//写文件的类
        int codec=VideoWriter::fourcc('M','J','P','G');//输出视频格式
        double fps=30.0;//输出视频帧数
        string name="output.mp4";//输出视频名称
        video>>frame;//读取视频一帧确定分辨率
        writer.open(name,codec,fps,frame.size(),1);


        while(1){
            video >> frame;  //读取每帧
            if (frame.empty()) {
                cout<<"Video Load Error!"<<endl;
                break;//读取失败直接退出
            }
        //预处理图像
            split(frame, channels); //通道分离
            cv::threshold(channels[color], binary, 220, 255, 0);//二值化颜色通道
            GaussianBlur(binary, Gaussian, Size(5, 5), 0);//滤波
            dilate(Gaussian, dilatee, element);//膨胀，把滤波得到的细灯条变宽
            findContours(dilatee, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);//轮廓检测
            vector<LightDescriptor> lightInfos;//创建一个灯条类的动态数组


        //筛选灯条
            for (int i = 0; i < contours.size(); i++) {
            // 求轮廓面积
                double area = contourArea(contours[i]);
            // 去除较小轮廓&fitEllipse的限制条件
                if (area < armored_plate_area || contours[i].size() <= 1)
                    continue;//太小了或者识别到的矩形亮块小于2块

            // 用椭圆拟合区域得到外接矩形（特殊的处理方式：因为灯条是椭圆型的，所以用椭圆去拟合轮廓，再直接获取旋转外接矩形即可）
                RotatedRect Light_Rec = fitEllipse(contours[i]);
            // 长宽比限制
                if (Light_Rec.size.height / Light_Rec.size.width > lightdescriptor_length_width_ratio[1]||Light_Rec.size.height / Light_Rec.size.width < lightdescriptor_length_width_ratio[0])
                    continue;
                lightInfos.push_back(LightDescriptor(Light_Rec));
            }
            //cout<<contours.size()<<endl;
        //灯条间进行匹配
            for (size_t i = 0; i < lightInfos.size(); i++) {
                for (size_t j = i + 1; (j < lightInfos.size()); j++) {
                    LightDescriptor& leftLight = lightInfos[i];
                    LightDescriptor& rightLight = lightInfos[j];

                    circle(frame,leftLight.center,1,Scalar(0,255,0),4);

                    float angle1=leftLight.angle;float angle2=rightLight.angle;
                    if(abs(angle1-angle2)>90){angle1=180-angle1;}
                    float angleGap_ = abs(angle1 - angle2)>90?(180-abs(angle1 - angle2)):abs(angle1 - angle2);//两个备选灯条的角度偏差
                    float Len = (leftLight.length + rightLight.length) / 2; //均长
                    float Wid = pow(pow((leftLight.center.x - rightLight.center.x), 2) + pow((leftLight.center.y - rightLight.center.y), 2), 0.5);//中心距离近似为宽度  
                    float len_wid_ratio=max(Len,Wid)/min(Len,Wid);
                    float angles =abs(atan(leftLight.center.y-rightLight.center.y)/ (leftLight.center.x-rightLight.center.x))/3.14159*180;
                    float angle=(angle1+angle2) / 2;
                    float delta;
                    if(abs(angles-angle)>90){delta=abs(180-angles-angle);}
                    else{delta=abs(angles-angle);}
                    float LenGap_ratio = abs(leftLight.length - rightLight.length) / max(leftLight.length, rightLight.length);//两备选灯条的长度偏差
                    float dis = Wid;
                    float meanLen = (leftLight.length + rightLight.length) / 2; //均长
        
                //匹配不通过的条件
                    if (
                    angleGap_ > angle_diff||
                    LenGap_ratio >length_diff||
                    len_wid_ratio<armored_plate_length_width_ratio[0]||
                    len_wid_ratio>armored_plate_length_width_ratio[1]||delta>80
                    ) {
                        continue;
                    }
                //绘制矩形
                    Point center = Point((leftLight.center.x + rightLight.center.x) / 2, (leftLight.center.y + rightLight.center.y) / 2);
                    RotatedRect rect = RotatedRect(center, Size(dis, meanLen), angle);
                    Point2f vertices[4];
                    rect.points(vertices);
                    for (int i = 0; i < 4; i++) {
                        line(frame, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255), 1);
                        circle(frame,center,3,Scalar(0,255,0),4);
                    }
                }
            }
            namedWindow("video2", WINDOW_FREERATIO);
            namedWindow("video3", WINDOW_FREERATIO);
            namedWindow("video1", WINDOW_FREERATIO);
            writer.write(frame);
            
            imshow("video1", frame);
            imshow("video2", channels[color]);
            imshow("video3", binary);
            if(waitKey(1)>=0){break;};
        }
        video.release();
        writer.release();
        destroyAllWindows();
    }
};
























int main()
{
    int color =0;
    int threshold=220;
    float lightdescriptor_length_width_ratio[2]={2,7};
    float angle_diff=3;
    float length_diff=0.1;
    float armored_plate_area=10;
    float armored_plate_length_width_ratio[2]={1,4};



    
    detector dec(color,threshold,lightdescriptor_length_width_ratio,angle_diff,length_diff,armored_plate_area,armored_plate_length_width_ratio);
    dec.detect();
    return 0;
}