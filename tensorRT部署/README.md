# tensorRT部署yolo模型
### 使用方式
```
./export XXX.onnx XXX.engine
./inference XXX(image file) (这里设定使用face.engine推理)
```
+ 得到相应的输出图片，用长方形框对人脸进行了标记。
### Tensor安装
+ 直接到NVIDIA官网找，下载deb格式的安装包。执行以下命令完成安装，报错则需先执行第4行。
```
os=”ubuntu1x04”
tag=”cudax.x-trt7.x.x.x-ga-yyyymmdd”
sudo dpkg -i nv-tensorrt-repo-${os}-${tag}_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-${tag}/7fa2af80.pub

sudo apt-get update
sudo apt-get install tensorrt

```

### 模型部署
+ 这里直接使用之前训练的只识别人脸的yolov5s模型，名为face.onnx，调用export程序将其转为face.engine引擎。也可以直接在yolov5s的仓库里调用yolo官方写的export.py将face.pt转为face.engine。
```
./export face.onnx face.engine
```
1. 载入engine引擎：
```c++
char *trtModelStream{ nullptr }; //trtModelStream的指针
size_t size{ 0 }; 

std::ifstream file("face.engine", std::ios::binary); 
if (file.good()) { 
        file.seekg(0, file.end); //运动到尾指针
        size = file.tellg(); //获取当前指针，也就得到了文件大小
        file.seekg(0, file.beg); //移动到头指针
        trtModelStream = new char[size]; //创建同样大小的文件流，初始化trtModelStream（char的大小是1byte，其实就是以二进制格式写入）
        assert(trtModelStream); //检查点，创建文件流失败则退出
        file.read(trtModelStream, size); //读取
        file.close(); //关闭
} 
```
+ 这里把二进制格式的.engine文件读入二进制文件流trtModelStream，将在下步载入ICudaEngine类中。


2. 创建createInferRuntime
```
Logger m_logger; 
IRuntime* runtime = createInferRuntime(m_logger); 
ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr); 
IExecutionContext* context = engine->createExecutionContext(); 

```
+ 这里ILogger类是抽象类，必须实现一个子类继承它来实例化，并且子类要求重写log函数。其实是个报告信息用的类，不想打印提示信息可以随便写。这里实现了一个子类Logger来实例化对象。
+ 之后创建一个runtime，顾名思义运行状态，它由m_logger报告信息。
+ 接着创建推理引擎，也就是把刚刚的二进制文件流trtModelStream写进来。
+ 创建一个IExecutionContext，直译是当前线程的上下文，作用是开一个线程。一个engine可以有多个executioncontext，并允许将同一套weights用于多个推理任务。
3. 分配显存
```c++
cudaMalloc(&buffers[inputIndex], batchSize * CHANNEL * IN_H * IN_W * sizeof(float)); 
cudaMalloc(&buffers[outputIndex], OUT * sizeof(float)); 
```
+ 这里分配了输入和输出数据的缓存区。

4. 搬数据到显存上并推理
```c++
// Create stream 
cudaStream_t stream; 
cudaStreamCreate(&stream); 

// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host 
cudaMemcpyAsync(buffers[inputIndex], input, batchSize * CHANNEL * IN_H * IN_W * sizeof(float), cudaMemcpyHostToDevice, stream); 
context.enqueue(batchSize, buffers, stream, nullptr); 
cudaMemcpyAsync(output, buffers[outputIndex], OUT * sizeof(float), cudaMemcpyDeviceToHost, stream); 
cudaStreamSynchronize(stream); 
```

+ 首先实例化一个cuda流，完成数据搬运功能。
+ 把输入数组搬到显存上。
+ 推理，这时输出数据被保存在buffer[1]里。
+ 把输出数据搬回内参上给cpu进一步操作。

5. 后续处理
+ 现在输出数据保存在output里，按照face.engine，它的大小为1 * 3600 * 6。现在可以依次读出3600个boudingbox的6个数据，进行非极大值抑制，然后完成绘图。

6. 效果图
+ ![](https://github.com/b-Acid/24-vision-lwh/blob/main/tensorRT%E9%83%A8%E7%BD%B2/out-1.png?raw=true)
