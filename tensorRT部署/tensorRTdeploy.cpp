#include <fstream> 
#include <iostream> 
#include <assert.h>
#include <NvInfer.h> 
#include <opencv2/opencv.hpp>


#define DEVICE 0
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0) 
 
using namespace nvinfer1; 
using namespace std;
using namespace cv;


class Logger:public  ILogger
{
	void log(Severity severity,const char* msg)  noexcept override
	{
		// if(severity != Severity::kINFO)
		// 	cout<<msg<<std::endl;	
                //懒得打印警告信息了
	}
}gLogger;

//box
typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
} BoxInfo;
static const int  classes=1;//类别
static const float NMS_THRESHOLD=0.3;//非极大值抑制阈值
static const float OBJ_THRESHOLD=0.6;//目标置信度阈值
static const int IN_H = 320; //输入大小
static const int IN_W = 320; 
static const int CHANNEL = 3;//输入通道 
static const int BATCH_SIZE = 1; //输入batch
static const long  OUT = 1*6300*6; //输出大小（6300个box，4个坐标加1个置信度加1个类别）

 
 

Mat processimg(string name,float*array)
{
        // 读取图片
        Mat image =imread(name,IMREAD_COLOR);

        // 改变图片大小为320*320
        resize(image, image,Size(IN_W, IN_H));

        // 归一化像素值
        Mat normalizedImage;
        image.convertTo(normalizedImage, CV_32F, 1.0 / 255.0);
        // 将像素Mat转换为一维数组   
        for (int c = 0; c < CHANNEL; c++)
        {
                for (int row = 0; row < IN_H; row++)
                {
                        for (int col = 0; col < IN_W; col++)
                        {
                               array[c * IN_H * IN_W + row * IN_H + col]=(normalizedImage.at<cv::Vec3f>(row, col)[c]); 
                        }
                }
        }
        return image;

}
//非极大值抑制
void nms(vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; }); // 降序排列
	vector<float> vArea(input_boxes.size());
	for (int i = 0; i < input_boxes.size(); ++i)
	{
		vArea[i] = (input_boxes[i].x2 - input_boxes[i].x1 + 1)
			* (input_boxes[i].y2 - input_boxes[i].y1 + 1);
	}
	// 全初始化为false，用来作为记录是否保留相应索引下pre_box的标志vector
	vector<bool> isSuppressed(input_boxes.size(), false);  
	for (int i = 0; i < input_boxes.size(); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < input_boxes.size(); ++j)
		{
			if (isSuppressed[j]) { continue; }
			float xx1 = max(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = max(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = min(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = min(input_boxes[i].y2, input_boxes[j].y2);
 
			float w = max(0.0f, xx2 - xx1 + 1);
			float h = max(0.0f, yy2 - yy1 + 1);
			float inter = w * h;	// 交集
			if(input_boxes[i].label == input_boxes[j].label)
			{
				float ovr = inter / (vArea[i] + vArea[j] - inter);  // 计算iou
				if (ovr >= NMS_THRESHOLD)
				{
					isSuppressed[j] = true;
				}
			}	
		}
	}
	// return post_nms;
	int idx_t = 0;
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize) 
{ 
        const ICudaEngine& engine = context.getEngine(); 
 
        // Pointers to input and output device buffers to pass to engine.
        // Engine requires exactly IEngine::getNbBindings() number of buffers. 
        assert(engine.getNbBindings() == 2); 
        void* buffers[2]; 

 
        // In order to bind the buffers, we need to know the names of the input and output tensors. 
        // Note that indices are guaranteed to be less than IEngine::getNbBindings() 
        const int inputIndex = 0;
        const int outputIndex = 1;
        assert(inputIndex==0);
 
        // Create GPU buffers on device 
        CHECK(cudaMalloc(&buffers[inputIndex], batchSize * CHANNEL * IN_H * IN_W * sizeof(float))); 
        CHECK(cudaMalloc(&buffers[outputIndex], batchSize * CHANNEL * IN_H * IN_W /4 * sizeof(float))); 
 
        // Create stream 
        cudaStream_t stream; 
        CHECK(cudaStreamCreate(&stream)); 
 
        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host 
        CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * CHANNEL * IN_H * IN_W * sizeof(float), cudaMemcpyHostToDevice, stream)); 
        context.enqueue(batchSize, buffers, stream, nullptr); 
        CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * CHANNEL * IN_H * IN_W / 4 * sizeof(float), cudaMemcpyDeviceToHost, stream)); 
        cudaStreamSynchronize(stream); 
 
        // Release stream and buffers 
        cudaStreamDestroy(stream); 
        CHECK(cudaFree(buffers[inputIndex])); 
        CHECK(cudaFree(buffers[outputIndex])); 
} 
 
int main(int argc, char** argv) 
{ 
        cout<<"Face detect"<<endl;
        clock_t startTime,endTime; //计算时间用的
        if(argv[1]==NULL)
        {cout<<"Parameter error:Please input file name!"<<endl;assert(argv[1]!= nullptr);}
        string filename=argv[1];
        // create a model using the API directly and serialize it to a stream 
        char *trtModelStream{ nullptr }; 
        size_t size{ 0 }; 
 
        std::ifstream file("face.engine", std::ios::binary); 
        if (file.good()) { 
                file.seekg(0, file.end); //运动到尾指针
                size = file.tellg(); //获取当前指针，也就得到了文件大小
                file.seekg(0, file.beg); //移动到头指针
                trtModelStream = new char[size]; //创建同样大小的文件流（char是1byte）
                assert(trtModelStream); //检查点，创建文件流失败则退出
                file.read(trtModelStream, size); //读取
                file.close(); //关闭
        } 
 
        Logger m_logger; 
        IRuntime* runtime = createInferRuntime(m_logger); 
        assert(runtime != nullptr); 
        ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr); 
        assert(engine != nullptr); 
        IExecutionContext* context = engine->createExecutionContext(); 
        assert(context != nullptr); 
 

        //read image and generate input data 
        float input_data[BATCH_SIZE * CHANNEL * IN_H * IN_W]; 
        startTime=clock();//计时，从预处理图片到绘制输出
        Mat image=processimg(filename,input_data);
        
        // Run inference 
        float prob[OUT]; 
        doInference(*context, input_data, prob, BATCH_SIZE); 

        // 非极大值抑制并绘图
	vector<BoxInfo> out;
        for (int i = 0; i < sizeof(prob)/sizeof(float); i+=classes+5) 
        {
                if(prob[i+4]>OBJ_THRESHOLD)
                        out.push_back(BoxInfo{prob[i]-prob[i+2]/2, prob[i+1]-prob[i+3]/2, prob[i]+prob[i+2]/2, prob[i+1]+prob[i+3]/2, prob[i+4], 1 });      
        }
        nms(out);
        cout<<"index\t"<<"x1\t"<<"y1\t"<<"x2\t"<<"y2\t"<<"score"<<endl;
        for (size_t i = 0; i < out.size(); i+=1)
        {
               cout<<i+1<<"\t"<<out[i].x1<<"\t"<<out[i].y1<<"\t"<<out[i].x2<<"\t"<<out[i].y2<<"\t"<<out[i].score<<endl;
               rectangle(image, Point(int(out[i].x1), int(out[i].y1)), Point(int(out[i].x2), int(out[i].y2)), Scalar(0, 0, 255), 2);
        }
        endTime=clock();
        cout<<"Use  "<<(double(endTime-startTime)/(CLOCKS_PER_SEC))<<"s"<<endl;
        imwrite("out-"+filename,image);
        cout<<"Out put image has been saved to  "<<"out-"+filename<<endl;



        //删内存
        context->destroy(); 
        engine->destroy(); 
        runtime->destroy(); 
        return 0; 
} 