# ONNXRUNTIME部署yolov5模型和装甲板检测模型
###
+ 使用说明
```
./classify filename(image)
./facedetect filename(video)
./yolodetect filename(video)
```
+ 编译时修改CMakeLists.txt里的onnxruntime的头文件和库的路径为自己电脑上的路径。
### ONNXRUNTIME安装
1. 这里直接选择官网下载源码编译，这里下载[onnxruntime1.8.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.8.0),找到assert里的source code点击下载。
2. 解压缩包，运行build.sh，等待编译完成。
3. 记住文件夹的路径，接下来的c++部署里要使用onnxruntime库。
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
    + 可能会弹出警告，提示大量目标的大小宽度小于5像素，这个直接不用管。
      

### 模型的转化
1. 在yolov5仓库中运行export.py，指定参数--weights(刚刚训练得到的权重文件)，--include onnx(保存为onnx文件)，--simplify(简化模型优化节点)。
  ```
  source activate xxx (激活安装好了yolo需要的包的python核心)
  python export.py --weights best.pt --include onnx --simplify
  ```
2. 得到的.onnx文件就可以调用onnxruntime的api在c++中部署了。

### 模型的部署
+ 本例中创建了两个可执行文件，分别调用了原生的yolov5s.onnx模型（识别80个类）和自己训练的best.onnx模型(只识别人脸),两者过程基本一致。
1. Configuration类
   
   ```c++
   struct Configuration
   {
   	public: 
   	float confThreshold; // Confidence threshold
   	float nmsThreshold;  // Non-maximum suppression threshold
   	float objThreshold;  //Object Confidence threshold
   	string modelpath; //模型路径
   }
   ```
   + 这个类指明了模型的路径和一些阈值。
    
2. yolov5类
   
   ```c++
   //yolov5类
   class YOLOv5
   {
   public:
   	YOLOv5(Configuration config);
   	void detect(Mat& frame);
   private:
   	float confThreshold;
   	float nmsThreshold;
   	float objThreshold;//三个配置量，也就是三个阈值
   	int inpWidth;
   	int inpHeight;//输入长宽
   	int nout;//输出的数据帧长度，对象中心xy，长宽wh，全局置信度（是一个对象的置信度），每个类别的置信度
   	int num_proposal;//探测到的类box数
   	int num_classes;//所有类别，也就是下面这些
   	string classes[1] = {"face"};
    
   	vector<float> input_image_;	// 输入图片
   	void normalize_(Mat img);		// 归一化函数
   	void nms(vector<BoxInfo>& input_boxes);  //非极大值抑制函数
   	Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);//resize图片为模型输入大小
    
   	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "yolov5-6.1"); // 初始化环境
   	Session *ort_session = nullptr;    // 初始化Session指针
   	SessionOptions sessionOptions = SessionOptions();  //初始化Session对象用的配置类
   	vector<char*> input_names;  // 定义一个字符指针vector
   	vector<char*> output_names; // 定义一个字符指针vector
   	vector<vector<int64_t>> input_node_dims; // >=1 inputs维度 
   	vector<vector<int64_t>> output_node_dims; // >=1 outputs维度
   };
   ```
   + 这步定义了模型运行的基本配置，包括使用Configuration类来初始化YOLOv5类，注释里有详细标明。
3. session类的配置
   ```c++
   string model_path = config.modelpath;//模型路径
 
   //CUDA设置
   OrtCUDAProviderOptions cuda_options{
          0,
          OrtCudnnConvAlgoSearch::EXHAUSTIVE,
          std::numeric_limits<size_t>::max()/10,
          0,
          true
      };
   sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
   sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);  //设置图优化类型
   ort_session = new Session(env, (const char*)model_path.c_str(), sessionOptions);//应用设置
   ```
4. 推理过程
   ```c++
   array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };
 
    //创建输入tensor
	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info,
                            input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
 
	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
   	float* pdata = ort_outputs[0].GetTensorMutableData<float>(); //输出流的头指针
   ```
   + 这步便得到了输出的数组了，在yolov5s原生模型中ort_outputs[0]是 1 * 25000 * 85  大小，在我的人脸识别模型中则为 1 * 6000 * 6 大小。第二维是所有预测的bounding-box的数量，第三维前两项是位置，  第三第四项是大小，第五项是置信度，后面是目标属于每类的分数。
   + 这里的pdata是一维float数组，一定要按照低维到高纬的顺序读取。读取到了数据后就可以非极大值抑制并画框了。
### 效果图
![](https://github.com/b-Acid/24-vision-lwh/blob/main/onnx%E9%83%A8%E7%BD%B2/example2.png?raw=true)
![](https://github.com/b-Acid/24-vision-lwh/blob/main/onnx%E9%83%A8%E7%BD%B2/output.png?raw=true)


### 用同样的方式部署装甲板识别模型
+ 只需更改模型的输入输出维度和大小即可,具体改动见classify.cpp。效果如下：
  ![](https://github.com/b-Acid/24-vision-lwh/blob/main/onnx%E9%83%A8%E7%BD%B2/example1.png?raw=true)
