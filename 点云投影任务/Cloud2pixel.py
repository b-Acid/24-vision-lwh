import pandas as pd
import cv2
import numpy as np
import math as mt



w=1920
h=1080

CameraExtrinsicMat=np.array(
    ([ -7.1907391850483116e-03, 1.0494953004635377e-02, 9.9991907134097757e-01, 1.0984281510814174e-01],
    [-9.9997142335005851e-01, 2.2580773589691017e-03,-7.2148159989590677e-03, -1.8261670813403203e-02],
    [-2.3336137706425064e-03, -9.9994237686382270e-01,1.0478415848689249e-02, 1.7323651488230618e-01],
    [ 0., 0., 0., 1. ]))
CameraMat=np.array((
    [ 1.3859739625395162e+03, 0, 9.3622464596653492e+02],
    [0,1.3815353250336800e+03, 4.9459467170828475e+02],
    [ 0, 0, 1 ]))
distortion_coefficients=np.array([ 7.0444095385902794e-02, -1.8010798300183417e-01,
       -7.7001990711544465e-03, -2.2524968464184810e-03,
       1.4838608095798808e-01 ])

for p in ["one","two","three","four"]:
    for j in range(5):
        #读取
        file="智能感知激光雷达任务数据/"+p+"/cloud_"+str(j)+".csv"
        index=[]
        for i in range(len(file)):
            if file[i] == '/':
                index.append(i-len(file)+1)#多文件夹处理

        name=file[index[0]:-4]
        data=pd.read_csv(file).to_numpy()
        data=np.column_stack((data,np.ones([data.shape[0],1])))#加一维归一化，用于矩阵运算
        POINTS=np.copy(data[:,0:2])#记录坐标
        DISTANCE=np.copy(data[:,0])#记录距离


        #坐标变换
        for i in range(data.shape[0]):
            temp=(data[i,:]).T#点坐标转列向量
            point=np.matmul(temp,CameraExtrinsicMat)#矩阵计算相机坐标
            DISTANCE[i]=mt.sqrt(point[0]**2+point[1]**2+point[2]**2)#计算深度
            point=np.matmul(point[0:3],CameraMat)/point[2]#矩阵计算像素坐标
            POINTS[i,:]=point[0:2]


        #选择像素框内的点
        x = POINTS[:,0]
        y = POINTS[:,1]

        temp=np.copy(x)
        x = x[np.where((temp>-w/2)&(temp<w/2))]
        y = y[np.where((temp>-w/2)&(temp<w/2))]
        DISTANCE=DISTANCE[np.where((temp>-w/2)&(temp<w/2))]

        temp=np.copy(y)
        x = x[np.where((temp>-h/2)&(temp<h/2))]
        y = y[np.where((temp>-h/2)&(temp<h/2))]
        DISTANCE=DISTANCE[np.where((temp>-h/2)&(temp<h/2))]



        #不同深度绘制不同大小的点
        maxd=np.max(DISTANCE)
        mind=np.min(DISTANCE)
        cut=(maxd-mind)/4
        PixPoint=np.column_stack((x,y,DISTANCE))
        size=0

        
        #输出图片
        img=np.zeros((h,w,1),np.uint8)
        for i in range(PixPoint.shape[0]):
            if PixPoint[i,2]<mind+cut:
                size=0
            if mind+cut<=PixPoint[i,2]<mind+2*cut:
                size=1
            if mind+2*cut<=PixPoint[i,2]<mind+3*cut:
                size=2
            if mind+3*cut<=PixPoint[i,2]:
                size=3

            cv2.circle(img, (int(PixPoint[i,0]+w/2),int(PixPoint[i,1]+h/2)), size, (255, 255, 255), -1)#画点，其实就是实心圆
        cv2.imwrite("outputs/"+name+".jpg",img)
print("Done!")
