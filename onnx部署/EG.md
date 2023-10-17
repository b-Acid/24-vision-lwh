
```C++
#include <fstream> 
#include <iostream> 
 
#include <NvInfer.h> 
#include <../samples/common/logger.h> 
 
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
using namespace sample; 
 
const char* IN_NAME = "input"; 
const char* OUT_NAME = "output"; 
static const int IN_H = 224; 
static const int IN_W = 224; 
static const int BATCH_SIZE = 1; 
static const int EXPLICIT_BATCH = 1 << (int)(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH); 
 
 
void doInference(IExecutionContext& context, float* input, float* output, int batchSize) 
{ 
        const ICudaEngine& engine = context.getEngine(); 
 
        // Pointers to input and output device buffers to pass to engine. 
        // Engine requires exactly IEngine::getNbBindings() number of buffers. 
        assert(engine.getNbBindings() == 2); 
        void* buffers[2]; 
 
        // In order to bind the buffers, we need to know the names of the input and output tensors. 
        // Note that indices are guaranteed to be less than IEngine::getNbBindings() 
        const int inputIndex = engine.getBindingIndex(IN_NAME); 
        const int outputIndex = engine.getBindingIndex(OUT_NAME); 
 
        // Create GPU buffers on device 
        CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * IN_H * IN_W * sizeof(float))); 
        CHECK(cudaMalloc(&buffers[outputIndex], batchSize * 3 * IN_H * IN_W /4 * sizeof(float))); 
 
        // Create stream 
        cudaStream_t stream; 
        CHECK(cudaStreamCreate(&stream)); 
 
        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host 
        CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * IN_H * IN_W * sizeof(float), cudaMemcpyHostToDevice, stream)); 
        context.enqueue(batchSize, buffers, stream, nullptr); 
        CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * 3 * IN_H * IN_W / 4 * sizeof(float), cudaMemcpyDeviceToHost, stream)); 
        cudaStreamSynchronize(stream); 
 
        // Release stream and buffers 
        cudaStreamDestroy(stream); 
        CHECK(cudaFree(buffers[inputIndex])); 
        CHECK(cudaFree(buffers[outputIndex])); 
} 
 
int main(int argc, char** argv) 
{ 
        // create a model using the API directly and serialize it to a stream 
        char *trtModelStream{ nullptr }; 
        size_t size{ 0 }; 
 
        std::ifstream file("model.engine", std::ios::binary); 
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
 
        // generate input data 
        float data[BATCH_SIZE * 3 * IN_H * IN_W]; 
        for (int i = 0; i < BATCH_SIZE * 3 * IN_H * IN_W; i++) 
                data[i] = 1; 
 
        // Run inference 
        float prob[BATCH_SIZE * 3 * IN_H * IN_W /4]; 
        doInference(*context, data, prob, BATCH_SIZE); 
        
        // 验证的输入数据为1，所以输出结果也为1 （因此，这里的验证结果与python的不同）
	    cout << *prob << endl;
 
        // Destroy the engine 
        context->destroy(); 
        engine->destroy(); 
        runtime->destroy(); 
        return 0; 
} 
```
