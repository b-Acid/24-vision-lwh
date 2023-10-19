#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <cuda_provider_factory.h>   // 提供cuda加速
#include <onnxruntime_cxx_api.h>	 // C或c++的api
#include<vector>

// 命名空间
using namespace std;
using namespace cv;
using namespace Ort;

 
//MYNET类
class MYNET
{
public:
	MYNET(string model_path);
	int detect(Mat& frame);
private:
	int inpWidth;
	int inpHeight;//输入长宽
	int num_classes;//所有类别，也就是下面这些
	string classes[6] = {"0","1","2","3","4","5"};
 
	vector<float> input_image_;		// 输入图片
	void normalize_(Mat img);		// 归一化函数
	Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);//resize图片为模型输入大小
 
	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "MY-NET"); // 初始化环境
	Session *ort_session = nullptr;    // 初始化Session指针选项
	SessionOptions sessionOptions = SessionOptions();  //初始化Session对象用的配置类
	vector<char*> input_names;  // 定义一个字符指针vector
	vector<char*> output_names; // 定义一个字符指针vector
	vector<vector<int64_t>> input_node_dims; // >=1 inputs维度 
	vector<vector<int64_t>> output_node_dims; // >=1 outputs维度
};

//构造函数
MYNET::MYNET(string model_path)
{
	this->num_classes = sizeof(this->classes)/sizeof(this->classes[0]);  // 类别数量
	this->inpHeight = 80;
	this->inpWidth = 80;//输入图片大小，可以调小点加速，也可以设置为原图大小
	
	string model = model_path;//模型路径
 
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
	ort_session = new Session(env, (const char*)model.c_str(), sessionOptions);//应用设置
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
	this->num_classes = output_node_dims[0][1];  //输出维度
 
}


void MYNET::normalize_(Mat img)
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
				float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];  // Mat里的ptr函数访问任意一行像素的首地址,2-c:表示rgb
				this->input_image_[c * row * col + i * col + j] = 2*pix / 255.0-1;
 
			}
		}
	}
}

//resize输入大小，yolo要求是正方形
Mat MYNET::resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (srch != srcw) {
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


int MYNET::detect(Mat& frame)
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
	
	float* pdata = ort_outputs[0].GetTensorMutableData<float>(); 

    int max= 0;
    float score=pdata[0];

    for(int i=1;i<num_classes;i++)
    {
        if(pdata[i]>score)
        {
            max=i;score=pdata[i];
        }    
    }
    return max;
	
}

int main(int argc,char **argv)
{
    clock_t startTime,endTime,STime,ETime; //计算时间用的
	string filename=argv[1];
    Mat img=imread(filename);
    MYNET my_model("ArmoClassificacion.onnx");
    cout<<"这是数字为"<<my_model.detect(img)<<"的装甲板"<<endl;
    return 0;

}