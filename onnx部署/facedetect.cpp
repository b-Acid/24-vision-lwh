#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_provider_factory.h>   // 提供cuda加速
#include <onnxruntime_cxx_api.h>	 // C或c++的api
 
//宏
#define SHOW_IMG 0

// 命名空间
using namespace std;
using namespace cv;
using namespace Ort;
 
// 自定义配置结构体
struct Configuration
{
	public: 
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	float objThreshold;  //Object Confidence threshold
	string modelpath;
};
 
// 定义BoxInfo结构类型，画框用
typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	int label;
} BoxInfo;
 
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
 
	const bool keep_ratio = true;
	vector<float> input_image_;		// 输入图片
	void normalize_(Mat img);		// 归一化函数
	void nms(vector<BoxInfo>& input_boxes);  //非极大值抑制函数
	Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);//resize图片为模型输入大小
 
	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "yolov5-6.1"); // 初始化环境
	Session *ort_session = nullptr;    // 初始化Session指针选项
	SessionOptions sessionOptions = SessionOptions();  //初始化Session对象用的配置类
	vector<char*> input_names;  // 定义一个字符指针vector
	vector<char*> output_names; // 定义一个字符指针vector
	vector<vector<int64_t>> input_node_dims; // >=1 inputs维度 
	vector<vector<int64_t>> output_node_dims; // >=1 outputs维度
};
 


//构造函数
YOLOv5::YOLOv5(Configuration config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	this->objThreshold = config.objThreshold;
	this->num_classes = sizeof(this->classes)/sizeof(this->classes[0]);  // 类别数量
	this->inpHeight = 320;
	this->inpWidth = 320;//输入图片大小，可以调小点加速，也可以设置为原图大小
	
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
	size_t numInputNodes = ort_session->GetInputCount();  //输入输出节点数量                         
	size_t numOutputNodes = ort_session->GetOutputCount(); //输出输出节点数量  
	AllocatorWithDefaultOptions allocator;   // 配置输入输出节点内存
	for (int i = 0; i < numInputNodes; i++)
	{
		input_names.push_back(ort_session->GetInputName(i, allocator));		// 内存
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);   // 类型
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();   
		auto input_dims = input_tensor_info.GetShape();    // 输入shape
		input_node_dims.push_back(input_dims);	// 保存
	}
	for (int i = 0; i < numOutputNodes; i++)
	{
		output_names.push_back(ort_session->GetOutputName(i, allocator));
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		output_node_dims.push_back(output_dims);
	}
	this->inpHeight = input_node_dims[0][2];	//
	this->inpWidth = input_node_dims[0][3];		//长宽
	this->nout = output_node_dims[0][2];      //输出的数据帧长度
	this->num_proposal = output_node_dims[0][1];  // 预测到的BOX数
 
}


//resize输入大小，yolo要求是正方形
Mat YOLOv5::resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 114);
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 114);
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;
}
 

//输入中心归一化
void YOLOv5::normalize_(Mat img)
{
	//    img.convertTo(img, CV_32F);
	int row = img.rows;
	int col = img.cols;
	this->input_image_.resize(row * col * img.channels());  // vector大小
	for (int c = 0; c < 3; c++)  // bgr
	{
		for (int i = 0; i < row; i++)  // 行
		{
			for (int j = 0; j < col; j++)  // 列
			{
 
				this->input_image_[c * row * col + i * col + j] = img.ptr<uchar>(i)[j * 3 + 2 - c] / 255.0;// Mat里的ptr函数访问任意一行像素的首地址,2-c:表示rgb
 
			}
		}
	}
}
 


//非极大值抑制
void YOLOv5::nms(vector<BoxInfo>& input_boxes)
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
				if (ovr >= this->nmsThreshold)
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
 


//探测，主要是运行模型并绘制图像
void YOLOv5::detect(Mat& frame)
{
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);//resize同时会把上面四个量初始化为frame的大小
	this->normalize_(dstimg);


	// 定义一个输入矩阵，int64_t是下面作为输入参数时的类型
	array<int64_t, 4> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };
 
    //创建输入tensor
	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
 
	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());   // 开始推理
	
	
	
	
	//按阈值输出并进行非极大值抑制
	vector<BoxInfo> generate_boxes;  // BoxInfo自定义的结构体
	float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;
	float* pdata = ort_outputs[0].GetTensorMutableData<float>(); // GetTensorMutableData
	for(int i = 0; i < num_proposal; ++i) // 遍历所有的num_pre_boxes
	{
		int index = i * nout;      // 一个数据帧长度为nout
		float obj_conf = pdata[index + 4];  // 全局置信度分数（第一次筛选）
		if (obj_conf > this->objThreshold)  // 大于全局阈值
		{
			int class_idx = 0;
			float max_class_socre = 0;
			for (int k = 0; k < this->num_classes; ++k)
			{
				if (pdata[k + index + 5] > max_class_socre)
				{
					max_class_socre = pdata[k + index + 5];
					class_idx = k;
				}
			}
			max_class_socre *= obj_conf;   // 最大的类别置信度


			if (max_class_socre > this->confThreshold) // 再次筛选
			{ 
				float cx = pdata[index];  //x
				float cy = pdata[index+1];  //y
				float w = pdata[index+2];  //w
				float h = pdata[index+3];  //h
 
				float xmin = (cx - padw - 0.5 * w)*ratiow;
				float ymin = (cy - padh - 0.5 * h)*ratioh;
				float xmax = (cx - padw + 0.5 * w)*ratiow;
				float ymax = (cy - padh + 0.5 * h)*ratioh;
 
				generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, max_class_socre, class_idx });
			}
		}
	}
 
	nms(generate_boxes);//非极大值抑制


	//绘制
	for (size_t i = 0; i < generate_boxes.size(); ++i)
	{
		int xmin = int(generate_boxes[i].x1);
		int ymin = int(generate_boxes[i].y1);
		rectangle(frame, Point(xmin, ymin), Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), Scalar(0, 0, 255), 2);
		string label = format("%.2f", generate_boxes[i].score);
		label = this->classes[generate_boxes[i].label] + ":" + label;
		putText(frame, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
	}
}
 
int main(int argc,char **argv)
{
	clock_t startTime,endTime,STime,ETime; //计算时间用的
	string filename=argv[1];
	VideoCapture cap(filename);
	Configuration yolo_nets = { 0.3, 0.5, 0.6,"best.onnx" };
	YOLOv5 yolo_model(yolo_nets);
	Mat srcimg ;
	int Fps;
	string s;
	if (SHOW_IMG)
	{
		namedWindow("rusult", WINDOW_FREERATIO);
	}
	
	

	int framecount=cap.get(CAP_PROP_FRAME_COUNT);

	VideoWriter writer;//写文件的类
        int codec=VideoWriter::fourcc('m','p','4','v');//输出视频格式
        double fps=30;//输出视频帧数
        string name="\""+filename+"\"-face-output.mp4";//输出视频名称
        cap>>srcimg;//读取视频一帧确定分辨率
        int frameH    = (int) srcimg.rows;
	    int frameW    = (int) srcimg.cols;
        cout<<"Video size: "<<frameH<<" * "<<frameW<<endl;
		cout<<"Frame counts: "<<framecount<<endl;
		cout<<"Processing......"<<endl;
        writer.open(name,codec,fps,srcimg.size(),1);

	STime = clock();//计时开始
	cout.precision(3);
	cout.width(50);
	cout.fill(' ');
	long i=2;
	while(1)
	{
		cap>>srcimg;
		if(srcimg.empty())break;
		startTime = clock();//计时开始	
		yolo_model.detect(srcimg);
		endTime = clock();//计时结束
		Fps = (int)(CLOCKS_PER_SEC / (double)(endTime - startTime));
        s = to_string(Fps)+"fps";
        putText(srcimg, s, Point(0, 30), FONT_HERSHEY_COMPLEX, 1.0, Scalar(255, 255, 255), 1, 8);
		s = to_string((float)(1)/Fps);+"s";
        putText(srcimg, s, Point(0, 60), FONT_HERSHEY_COMPLEX, 1.0, Scalar(255, 255, 255), 1, 8);
		if(SHOW_IMG)
		{
			imshow("rusult",srcimg);
			if(waitKey(1)==27)break;
		}
		writer.write(srcimg);
		if(i%(int)(framecount/30)==0)
		{
			
			for(int j=0;j<70;j++)
				cout<<"\b";
			cout<<"[";
			for(int j=0;j<int(30*i/framecount);j++)
				cout<<"=";
			for(int j=0;j<30-int(30*i/framecount);j++)
				cout<<" ";
			cout<<"]"<<100*float(i)/framecount<<"%";
			ETime=clock();
			cout<<"    Task ends in "<<(double(ETime-STime)/(CLOCKS_PER_SEC))/i*(framecount-i)<<"s    ";
			fflush(stdout);
		}
		i++;
	}
	ETime = clock();//计时结束
	for(int j=0;j<60;j++)
		cout<<"\b";
	cout<<endl<<"All task done in "<<double(ETime-STime)/(CLOCKS_PER_SEC)<<" seconds!"<<endl;
	cout<<"File has been saved to \"output.mp4\""<<endl;
	cap.release();
	writer.release();
	destroyAllWindows();
	return 0;
}

