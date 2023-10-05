#include "stdio.h"
#include<iostream> 
#include <cmath> 
#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
#include<string>
#include<ctime>
using namespace std;
using namespace cv;


#define RED 0
#define BLUE 1
#define SHOW_BIN_IMAGE 0

double getDistance (Point2f point1, Point2f point2)
{
    double distance = sqrtf(powf((point1.x - point2.x),2) + powf((point1.y - point2.y),2));
    return distance;
}

class LightDescriptor//灯管类
{	    
public:float width, length, angle, area;
    Point2f center,points[4],point2[2];
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
        light.points(points);
        Point2f temp1=(points[0]+points[1])/2;
        Point2f temp2=(points[2]+points[1])/2;
        double dis1=getDistance(temp1,center);
        double dis2=getDistance(temp2,center);
        if(dis1>=dis2)
        {
            if(temp1.y<center.y)
            {
                point2[0]=temp1;
                point2[1]=2*center-temp1; 
            }
            else
            {
                point2[1]=temp1;
                point2[0]=2*center-temp1; 
            } 
        }
        else
        {
            if(temp2.y<center.y)
            {
                point2[0]=temp2;
                point2[1]=2*center-temp2; 
            }
            else
            {
                point2[1]=temp2;
                point2[0]=2*center-temp2; 
            } 
        }
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
        this->lightescriptor_area=lightescriptor_area;
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

        Mat frame, channels[3], colorsplit,binary, Gaussian, dilatee;
        

        VideoWriter writer;//写文件的类
        int codec=VideoWriter::fourcc('M','J','P','G');//输出视频格式
        double fps=15.0;//输出视频帧数
        string name="output.avi";//输出视频名称
        video>>frame;//读取视频一帧确定分辨率
        int frameH    = (int) frame.rows;
	    int frameW    = (int) frame.cols; 
        Point2d a = Point2d(0,frameH/2);
        Point2d b = Point2d(frameW,frameH/2);
        cout<<frameH<<frameW<<endl;
        writer.open(name,codec,fps,frame.size(),1);


        vector<Point3d> model_points;//pnp解算世界坐标
        model_points.push_back(Point3d(-69.0f, +26.0f, 0)); // 左上角
        model_points.push_back(Point3d(+69.0f, +26.0f, 0));
        model_points.push_back(Point3d(+69.0f, -26.0f, 0));
        model_points.push_back(Point3d(-69.0f, -26.0f, 0)); // 右下角


        Mat camera_matrix = (Mat_<double>(3, 3) << 1000,0,frameW/2,0,1000,frameH/2,0,0,1);//相机参数
        Mat dist_coeffs = (Mat_<double>(5, 1) <<0,0,0,0,0);//畸变系数
        Mat rotation_vector;//旋转向量
	    Mat translation_vector;// 平移向量
        vector<Point3f> pointsets;
        pointsets.push_back(Point3f(20,0, 0));
        pointsets.push_back(Point3f(0,20,0));
        pointsets.push_back(Point3f(0,0,20));
        vector<Point2f> imagepoints;
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        int Fps;
        

        
        

        namedWindow("video", WINDOW_FREERATIO);
        if(SHOW_BIN_IMAGE)namedWindow("video2", WINDOW_FREERATIO);
        while(1){
            clock_t start = clock();
            video >> frame;  //读取每帧
            if (frame.empty()) {
                cout<<"Video Load Done!"<<endl;
                break;//读取完退出
            }
        //预处理图像
            split(frame, channels); //通道分离
            //colorsplit=channels[color];
            if(color==BLUE)
                colorsplit=channels[0];
            else
                colorsplit=channels[2];
            cv::threshold(colorsplit, binary, threshold, 255, 0);//二值化颜色通道
            GaussianBlur(binary, Gaussian, Size(5, 5), 0);//滤波
            Mat element = getStructuringElement(MORPH_RECT, Size(9,9));
            dilate(Gaussian, dilatee, element);//膨胀，把滤波得到的细灯条变宽
            Mat element1 = getStructuringElement(MORPH_RECT, Size(5, 5));//设置内核1
            Mat element2 = getStructuringElement(MORPH_RECT, Size(5, 5));//设置内核2
            morphologyEx(dilatee, dilatee, MORPH_OPEN, element1);//开运算(使图形明显)
            floodFill(dilatee, Point(0, 0), Scalar(0));//漫水法
            morphologyEx(dilatee, dilatee, MORPH_CLOSE, element2);//闭运算(减少图形数量)
            findContours(dilatee, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);//轮廓检测
            sort(contours.begin(), contours.end(), [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                cv::Rect rectA = cv::boundingRect(a);
                cv::Rect rectB = cv::boundingRect(b);
                return rectA.x < rectB.x;
            });
            vector<LightDescriptor> lightInfos;//创建一个灯条类的动态数组


        //筛选灯条
            for (int i = 0; i < contours.size(); i++) {
            // 求轮廓面积
                double area = contourArea(contours[i]);
            // 去除较小轮廓&fitEllipse的限制条件
                if (area < lightescriptor_area || contours[i].size() <= 1)
                    continue;//太小了或者识别到的矩形亮块小于2块

            // 用椭圆拟合区域得到外接矩形（特殊的处理方式：因为灯条是椭圆型的，所以用椭圆去拟合轮廓，再直接获取旋转外接矩形即可）
                RotatedRect Light_Rec = fitEllipse(contours[i]);
            // 长宽比限制
                if (Light_Rec.size.height / Light_Rec.size.width > lightdescriptor_length_width_ratio[1]||Light_Rec.size.height / Light_Rec.size.width < lightdescriptor_length_width_ratio[0])
                    continue;
                lightInfos.push_back(LightDescriptor(Light_Rec));
            }



        //灯条间进行匹配
            for (size_t i = 0; i < lightInfos.size(); i++) {
                for (size_t j = i+1; j < lightInfos.size(); j++) {
                    LightDescriptor& leftLight = lightInfos[i];
                    LightDescriptor& rightLight = lightInfos[j];
                    Point2f vertices[4];//四个角点
                    vertices[0]=leftLight.point2[0];                   
                    vertices[1]=rightLight.point2[0];
                    vertices[2]=rightLight.point2[1];
                    vertices[3]=leftLight.point2[1];
                    circle(frame,leftLight.center,0.5,Scalar(0,255,0),3);

                    float angle1=leftLight.angle;float angle2=rightLight.angle;
                    if(abs(angle1-angle2)>90){angle1=180-angle1;}
                    float angleGap_ = abs(angle1 - angle2)>90?(180-abs(angle1 - angle2)):abs(angle1 - angle2);//两个备选灯条的角度偏差
                    float Len = (leftLight.length + rightLight.length) / 2; //均长
                    float Wid = pow(pow((leftLight.center.x - rightLight.center.x), 2) + pow((leftLight.center.y - rightLight.center.y), 2), 0.5);//中心距离近似为宽度  
                    float len_wid_ratio=Wid/Len;
                    float angles =atan(leftLight.center.y-rightLight.center.y)/ (leftLight.center.x-rightLight.center.x)<0?180+atan(leftLight.center.y-rightLight.center.y)/ (leftLight.center.x-rightLight.center.x)/3.14159*180:atan(leftLight.center.y-rightLight.center.y)/ (leftLight.center.x-rightLight.center.x)/3.14159*180;
                    float angle=(angle1+angle2) / 2;
                    float delta;
                    if(abs(angles-angle)>90){delta=90-abs(180-angles-angle);}
                    else{delta=90-abs(angles-angle);}
                    float LenGap_ratio = abs(leftLight.length - rightLight.length) / max(leftLight.length, rightLight.length);//两备选灯条的长度偏差
                    float dis = Wid;
                    float meanLen = (leftLight.length + rightLight.length) / 2; //均长
        
                //匹配不通过的条件
                    if (angleGap_ > angle_diff||
                    LenGap_ratio >length_diff||
                    len_wid_ratio<armored_plate_length_width_ratio[0]
                    ||len_wid_ratio>armored_plate_length_width_ratio[1]||delta<50
                    ) {
                        
                        continue;
                    }
                //绘制四边形
                    putText(frame,string("1"), vertices[0], FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 255, 255), 1, 3);
                    putText(frame,string("2"), vertices[1], FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 255, 255), 1, 3);
                    putText(frame,string("3"), vertices[2], FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 255, 255), 1, 3);
                    putText(frame,string("4"), vertices[3], FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 255, 255), 1, 3);
                    Point center = Point((leftLight.center.x + rightLight.center.x) / 2, (leftLight.center.y + rightLight.center.y) / 2);
                    for (int i = 0; i < 4; i++) {
                        line(frame, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255), 1);    
                        circle(frame,vertices[i],0.5,Scalar(0,255,0),3);
                    }
                    circle(frame,center,0.5,Scalar(0,255,0),3);
                    line(frame,leftLight.center,rightLight.center, Scalar(0, 0, 255), 1);

                    vector<Point2d> image_points;//pnp解算像素坐标
                    image_points.push_back(Point2d(vertices[0]));
                    image_points.push_back(Point2d(vertices[1]));
                    image_points.push_back(Point2d(vertices[2]));
                    image_points.push_back(Point2d(vertices[3]));
                    solvePnP(model_points, image_points, camera_matrix, dist_coeffs, rotation_vector, translation_vector);
                    projectPoints(pointsets,rotation_vector,translation_vector,camera_matrix, dist_coeffs,imagepoints);
                    //cout<<image_points[0]<<endl;
                    line(frame, center, imagepoints[0], Scalar(255, 0, 0), 1);  
                    line(frame, center, imagepoints[1], Scalar(0, 255, 0), 1);   
                    line(frame, center, imagepoints[2], Scalar(0, 0, 255), 1);    
                    Mat Rvec;
                    Mat_<float> Tvec;
                    rotation_vector.convertTo(Rvec, CV_32F);  // 旋转向量转换格式
                    translation_vector.convertTo(Tvec, CV_32F); // 平移向量转换格式 

                    Mat_<float> rotMat(3, 3);
                    Rodrigues(Rvec, rotMat);
                    // 旋转向量转成旋转矩阵

                    Mat P_oc;
                    P_oc = rotMat.inv() * Tvec;
                    // 求解相机的世界坐标，得出p_oc的第三个元素即相机到物体的距离，单位是mm
                    float Dis=(pow(pow(P_oc.at<float>(0,0), 2) + pow(P_oc.at<float>(0,1), 2)+pow(P_oc.at<float>(0,3), 2), 0.5)*10)/100;
                    string s = to_string(int(Dis))+"."+to_string(int(Dis*10-int(Dis)*10))+"cm";
                    putText(frame,s, (vertices[0]+vertices[3])/2, FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 255, 255), 1, 3);

                    float angel=(atan(P_oc.at<float>(0,1)/P_oc.at<float>(0,2)))/3.14*180;
                    string s2 = to_string(int(angel))+"."+to_string(int(abs(angel*10-int(angel)*10)))+"Du";
                    putText(frame,s2, a, FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 255, 255), 1, 3);

                }
            }

            
           



        //计时并打印信息
            clock_t end = clock();
            Fps = (int)(CLOCKS_PER_SEC / (double)(end - start)  );
		    //cout <<"   "<< Fps << "帧" << endl;
            string s = to_string(Fps)+"fps";
            putText(frame, s, Point(0, 25), FONT_HERSHEY_COMPLEX, 1.0, Scalar(255, 255, 255), 1, 8);
            line(frame,a,b, Scalar(0, 0, 255), 1);

            
            writer.write(frame);
            
            imshow("video", frame);
            if(SHOW_BIN_IMAGE)imshow("video2", dilatee);
            if(waitKey(30)==27){break;};
            
            
        }
        video.release();
        writer.release();
        destroyAllWindows();
    }
};











int main()
{
    int color =0;
    int threshold=140;
    float lightdescriptor_length_width_ratio[2]={1,10};
    float angle_diff=2;
    float length_diff=0.2;
    float lightescriptor_area=10;
    float armored_plate_length_width_ratio[2]={1,5};



    
    detector dec(color,threshold,lightdescriptor_length_width_ratio,angle_diff,length_diff,lightescriptor_area,armored_plate_length_width_ratio,"testvideo.avi");
    dec.detect();
    return 0;
}