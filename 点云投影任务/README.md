# 点云投影任务

### 原理
+ 相机外参是一个4×4矩阵
  
$$
CameraExtrinsicMat=\left[
 \begin{matrix}
    --0.00719 & 0.0104 & 1 & 0.1 \\
   -1 & -0.00225 & -0.00721 & -0.0182 \\
  -0.00223 & -1 & -0.0104 & -0.173\\
   0 & 0 & 0 & 1 
  \end{matrix}
  \right]
$$

+ 相机内参是一个3×3矩阵

$$
CameraMat=\left[
 \begin{matrix}
  1385 & 0 & 936\\
  0 &  1385 & 494\\
   0 &  0 & 1 
  \end{matrix}
  \right] 
$$

+ 世界坐标下的点 $[Xw,Yw,Zw]$ 扩维后化为列向量 $[Xw,Yw,Zw,1]^T $，左乘外参矩阵 $CameraExtrinsicMat$ 得到扩维相机坐标 $[Xc,Yc,Zc,1]$。
+ 相机坐标 $[Xc,Yc,Zc]$ 左乘内参矩阵$ CameraMat $再除以z轴深度Zc得到像素矩阵 $[u,v,1]$。
+ 依据深度信息对不同的世界点描绘不同大小的像素点。
+ 效果见如下：
[](https://github.com/b-Acid/24-vision-lwh/blob/main/%E7%82%B9%E4%BA%91%E6%8A%95%E5%BD%B1%E4%BB%BB%E5%8A%A1/outputs/one/cloud_2.jpg?raw=true)
