#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <iomanip>
#include <string>
#include <fstream>
#include <map>
#include <dirent.h>
#include <cstring>

#include <hdf5/serial/H5Cpp.h>

int max_point_num = 18000;
// 读取测试数据
using namespace H5;
//using namespace std;
#include <curand_kernel.h>


// CUDA 核函数：用于插值生成新点
__global__ void interpolate_points(float* points, float* new_points, int num_existing_points, int num_points_needed) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_points_needed) {
        // 随机生成两个点的索引进行插值
        curandState state;
        curand_init(1234, idx, 0, &state);
        int index_1 = curand(&state) % num_existing_points;
        int index_2 = curand(&state) % num_existing_points;

        // 生成插值系数 alpha，介于 0 到 1 之间
        float alpha = curand_uniform(&state);

        // 进行插值操作
        for (int i = 0; i < 3; ++i) {
            new_points[idx * 3 + i] = points[index_1 * 3 + i] + alpha * (points[index_2 * 3 + i] - points[index_1 * 3 + i]);
        }
    }
}

// CUDA 核函数：用于随机采样点
__global__ void sample_points(float* points, float* sampled_points, int num_existing_points, int num_sampled_points) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < num_sampled_points) {
        // 随机选择点进行采样
        curandState state;
        curand_init(1234, idx, 0, &state);
        int index = curand(&state) % num_existing_points;
        for (int i = 0; i < 3; ++i) {
            sampled_points[idx * 3 + i] = points[index * 3 + i];
        }
    }
}

// 使用 CUDA 处理点云数据的函数
void process_point_cloud_data_cuda(float* h_points, int num_points, std::vector<std::vector<float>>& list_of_points) {
    if (num_points < 18000) {
        int num_points_needed = 18000 - num_points;
        float* d_points;
        float* d_new_points;

        // 分配 GPU 内存
        cudaMalloc(&d_points, num_points * 3 * sizeof(float));
        cudaMalloc(&d_new_points, num_points_needed * 3 * sizeof(float));

        // 复制数据到 GPU
        cudaMemcpy(d_points, h_points, num_points * 3 * sizeof(float), cudaMemcpyHostToDevice);

        // 定义 CUDA 配置
        int threads_per_block = 256;
        int blocks_per_grid = (num_points_needed + threads_per_block - 1) / threads_per_block;

        // 启动插值核函数
        interpolate_points<<<blocks_per_grid, threads_per_block>>>(d_points, d_new_points, num_points, num_points_needed);

        // 将新生成的点复制回 host 端
        float* h_new_points = new float[num_points_needed * 3];
        cudaMemcpy(h_new_points, d_new_points, num_points_needed * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        // 将原始点和新点合并
        std::vector<float> combined_points(h_points, h_points + num_points * 3);
        combined_points.insert(combined_points.end(), h_new_points, h_new_points + num_points_needed * 3);

        // 存储处理后的点云数据
        list_of_points.push_back(combined_points);

        // 释放 GPU 内存
        cudaFree(d_points);
        cudaFree(d_new_points);
        delete[] h_new_points;

    } else if (num_points > 18000) {
        float* d_points;
        float* d_sampled_points;

        // 分配 GPU 内存
        cudaMalloc(&d_points, num_points * 3 * sizeof(float));
        cudaMalloc(&d_sampled_points, 18000 * 3 * sizeof(float));

        // 复制数据到 GPU
        cudaMemcpy(d_points, h_points, num_points * 3 * sizeof(float), cudaMemcpyHostToDevice);

        // 定义 CUDA 配置
        int threads_per_block = 256;
        int blocks_per_grid = (18000 + threads_per_block - 1) / threads_per_block;

        // 启动随机采样核函数
        sample_points<<<blocks_per_grid, threads_per_block>>>(d_points, d_sampled_points, num_points, 18000);

        // 将采样后的点复制回 host 端
        float* h_sampled_points = new float[18000 * 3];
        cudaMemcpy(h_sampled_points, d_sampled_points, 18000 * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        // 存储采样后的点云数据
        std::vector<float> sampled_points(h_sampled_points, h_sampled_points + 18000 * 3);
        list_of_points.push_back(sampled_points);

        // 释放 GPU 内存
        cudaFree(d_points);
        cudaFree(d_sampled_points);
        delete[] h_sampled_points;

    } else {
        // 点数量正好为 18000，直接存储
        std::vector<float> exact_points(h_points, h_points + num_points * 3);
        list_of_points.push_back(exact_points);
    }
}

// 读取HDF5文件并调用CUDA处理函数
void read_h5_file(const std::string& file_path, std::vector<std::vector<float>>& list_of_points, std::vector<int>& list_of_labels) {
    try {
        // 打开 HDF5 文件
        H5File file(file_path, H5F_ACC_RDONLY);

        // 获取文件中的所有数据集
        hsize_t num_objs = file.getNumObjs();
        for (hsize_t i = 0; i < num_objs; ++i) {
            std::string dataset_name = file.getObjnameByIdx(i);

            // 打开每个数据集的 "points" 子数据集
            if (file.childObjType(dataset_name) == H5G_GROUP) {
                Group group = file.openGroup(dataset_name);
                if (H5Lexists(group.getId(), "points", H5P_DEFAULT) > 0) {
                    DataSet dataset = group.openDataSet("points");
                    DataSpace dataspace = dataset.getSpace();

                    // 获取数据集的维度
                    hsize_t dims[2];
                    dataspace.getSimpleExtentDims(dims, nullptr);
                    hsize_t num_points = dims[0];
                    hsize_t num_dims = dims[1];

                    // 读取点云数据
                    float* h_points = new float[num_points * num_dims];
                    dataset.read(h_points, PredType::NATIVE_FLOAT);

                    // 调用 CUDA 函数处理点云数据
                    process_point_cloud_data_cuda(h_points, num_points, list_of_points);

                    // 释放 host 端内存
                    delete[] h_points;

                    // 读取标签
                    if (H5Aexists(group.getId(), "label") > 0) {
                        Attribute label_attr = group.openAttribute("label");
                        int label;
                        label_attr.read(PredType::NATIVE_INT, &label);

                        // 存储标签
                        list_of_labels.push_back(label);
                    } else {
                        std::cerr << "Warning: No 'label' attribute found for group: " << dataset_name << std::endl;
                    }
                }
            }
        }
    } catch (FileIException& error) {
        error.printErrorStack();
    } catch (DataSetIException& error) {
        error.printErrorStack();
    } catch (DataSpaceIException& error) {
        error.printErrorStack();
    } catch (DataTypeIException& error) {
        error.printErrorStack();
    } catch (GroupIException& error) {
        error.printErrorStack();
    } catch (AttributeIException& error) {
        error.printErrorStack();
    }
}


// 获取目录中的所有 .txt 文件
std::vector<std::string> get_files_in_directory(const std::string& dir) {
    std::vector<std::string> files;
    DIR* dp;
    struct dirent* entry;
    if ((dp = opendir(dir.c_str())) != NULL) {
        while ((entry = readdir(dp)) != NULL) {
            std::string filename = entry->d_name;
            if (filename.find(".txt") != std::string::npos) {
                files.push_back(filename);
            }
        }
        closedir(dp);
    } else {
        perror("opendir");
    }
    return files;
}

// 读取 .txt 文件并转换为 std::vector<float>
std::vector<float> read_param(const std::string& filepath) {
    std::vector<float> data;
    std::ifstream file(filepath);
    if (file.is_open()) {
        float value;
        while (file >> value) {
            data.push_back(value);
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filepath << std::endl;
    }
    return data;
}

std::map<std::string, std::vector<float>> read_params(std::string dir) {
    // std::string dir = "."; // 当前目录
    std::map<std::string, std::vector<float>> params;

    // 获取目录中的所有 .txt 文件
    std::vector<std::string> param_files = get_files_in_directory(dir);
    for (const auto& file : param_files) {
        std::string filename = file.substr(0, file.find_last_of(".")); // 获取不带扩展名的文件名
        params[filename] = read_param(dir + "/" + file);
    }

    // // 访问参数时可以使用 params["conv1_weight"]
    // for (const auto& kv : params) {
    //     std::cout << "Key: " << kv.first << ", Values: ";
    //     // for (const auto& value : kv.second) {
    //     //     std::cout << value << " ";
    //     // }
    //     std::cout << std::endl;
    // }

    return params;
}

void Free_cudapointer(float* &pointer)
{
    if(pointer != nullptr)
    {
        cudaFree(pointer);
        pointer = nullptr;
    }
};

__global__ void conv1d_kernel(const float* d_input, const float* d_weights, const float* d_biases, float* d_output,
                            int in_channels, int out_channels, int input_length, int kernel_size) 
{

    int idx_row = (blockIdx.x * blockDim.x + threadIdx.x) / input_length;
    int idx_col = (blockIdx.x * blockDim.x + threadIdx.x) % input_length;

    if(idx_row < out_channels && idx_col < input_length)
    {
        // output的第matX行，matY列需要weights的x行和input的y列相乘。
        float value = 0.0f;
        for(int count=0; count<in_channels; ++count)
        {
            value += d_weights[in_channels * idx_row + count] * d_input[input_length * count + idx_col];
            //printf("matX %d, matY %d \n",matX, matY);
            //printf("inchannel %d, count %d , weights %f, input=%f \n",in_channels, count, d_weights[in_channels * matX + count], d_input[input_length * count + matY]);
        }

        //output[matX * input_length + matY] = max(value, 0.0); //内执行relu
        d_output[idx_row * input_length + idx_col] = value + d_biases[idx_row];
            
    }
    
}
__global__ void add_kernel(int input_length, float* d_input, float* d_add_source, float* d_output)
{
    int idx_global = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx_global < input_length)
        d_output[idx_global] = d_input[idx_global] + d_add_source[idx_global];

}

// 用于解决数据最开始是3个3个排列的。
__global__ void transpose_kernel(int input_length, float* d_input, float* d_output, float* d_output2)
{
    int idx_global = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_row = idx_global / input_length;
    int idx_col = idx_global % input_length;
    if(idx_row < 3 && idx_global < 3 * input_length)
    {
        d_output[idx_row * input_length + idx_col] = d_input[idx_col * 3 + idx_row];
        d_output2[idx_row * input_length + idx_col] = d_input[idx_col * 3 + idx_row];        
    }

}
// 修复后的 Conv1D 类定义
class Conv1D 
{
public:
    int in_channels, out_channels, kernel_size, input_length;  // 1d长度
    std::vector<float>weights, biases;
    float *h_input;
    float *d_input, *d_output, *d_weights, *d_biases; // 修复指针声明

    // 构造函数：初始化成员变量，分配内存
    Conv1D(int in_channels, int out_channels, int kernel_size, std::vector<float> weights, std::vector<float> biases)
    : in_channels(in_channels), out_channels(out_channels), kernel_size(kernel_size), weights(weights), biases(biases)
    {
        cudaError_t err;
        // 分配权重和偏置内存
        err = cudaMalloc(&d_weights, out_channels * in_channels * kernel_size * sizeof(float));
        if (err != cudaSuccess) 
        {
            std::cerr << "cudaMalloc Conv1D malloc: d_weights failed: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        
        err = cudaMalloc(&d_biases, out_channels * sizeof(float));
        if (err != cudaSuccess) 
        {
            std::cerr << "cudaMalloc d_biases failed: " << cudaGetErrorString(err) << std::endl;
            Free_cudapointer(d_weights);
            return;
        }

        // 检查 cudaMemcpy 调用是否成功
        err = cudaMemcpy(d_weights, weights.data(), out_channels * in_channels * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy Conv1D cpy d_weights failed: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        err = cudaMemcpy(d_biases, biases.data(), out_channels * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy d_biases failed: " << cudaGetErrorString(err) << std::endl;
            return;
        }
    }
    // 析构函数：释放 GPU 内存
    ~Conv1D() 
    {
        // vector 有自动内存回收机制不需要析构
        cudaFree(d_weights);
        cudaFree(d_biases);
    }

    // 前向传播函数：调用 CUDA 内核函数
    void forward_HostToHost(int input_length, float* h_input, float* h_output) 
    {   
        cudaError_t err;
        // 分配输入内存
        err = cudaMalloc(&d_input, input_length * in_channels * sizeof(float));
        if (err != cudaSuccess) 
        {
            std::cerr << "cudaMalloc d_input failed: " << cudaGetErrorString(err) << std::endl;
            Free_cudapointer(d_input);
            return;
        }

        // 分配输出内存
        err = cudaMalloc(&d_output, input_length * out_channels * sizeof(float));
        if (err != cudaSuccess) 
        {
            std::cerr << "cudaMalloc conv1d host2host d_output failed: " << cudaGetErrorString(err) << std::endl;
            Free_cudapointer(d_output);
            return;
        }

        cudaMemcpy(d_input, h_input, input_length * in_channels * sizeof(float), cudaMemcpyHostToDevice);

        dim3 numBlocks((out_channels * input_length + 511) / 512);
        dim3 ThreadsPerBlock(512);

        conv1d_kernel <<<numBlocks, ThreadsPerBlock>>> (d_input, d_weights, d_biases, d_output, in_channels, out_channels, input_length, kernel_size);
        //cudaDeviceSynchronize();
        
        cudaMemcpy(h_output, d_output, out_channels * input_length * sizeof(float), cudaMemcpyDeviceToHost);
        
        Free_cudapointer(d_input);
        Free_cudapointer(d_output);
    }

    void forward_HostToDevice(int input_length, float* h_input, float* d_output) 
    {   
        cudaError_t err;
        // 分配输入内存
        err = cudaMalloc(&d_input, input_length * in_channels * sizeof(float));
        if (err != cudaSuccess) 
        {
            std::cerr << "cudaMalloc d_input failed: " << cudaGetErrorString(err) << std::endl;
            Free_cudapointer(d_input);
            return;
        }

        cudaMemcpy(d_input, h_input, input_length * in_channels * sizeof(float), cudaMemcpyHostToDevice);

        dim3 numBlocks((out_channels * input_length + 511) / 512);
        dim3 ThreadsPerBlock(512);

        conv1d_kernel <<<numBlocks, ThreadsPerBlock>>> (d_input, d_weights, d_biases, d_output, in_channels, out_channels, input_length, kernel_size);
        //cudaDeviceSynchronize();
        
        //cudaMemcpy(h_output, d_output, out_channels * input_length * sizeof(float), cudaMemcpyDeviceToHost);
        
        Free_cudapointer(d_input);
        //Free_cudapointer(d_output);
    }

    void forward_DeviceToDevice(int input_length, float* d_input, float* d_output) 
    {   
        cudaError_t err;
        // 分配输入内存

        dim3 numBlocks((out_channels * input_length + 511) / 512);
        dim3 ThreadsPerBlock(512);

        conv1d_kernel <<<numBlocks, ThreadsPerBlock>>> (d_input, d_weights, d_biases, d_output, in_channels, out_channels, input_length, kernel_size);
        //cudaDeviceSynchronize();
        
        //cudaMemcpy(h_output, d_output, out_channels * input_length * sizeof(float), cudaMemcpyDeviceToHost);
        
        //Free_cudapointer(d_input);
        //Free_cudapointer(d_output);
    }

    void forward_DeviceToHost(int input_length, float* d_input, float* h_output) 
    {   
        cudaError_t err;
        // 分配输入内存
        //float *d_output_device2host;
        Free_cudapointer(d_output);
        err = cudaMalloc(&d_output, input_length * out_channels * sizeof(float));
        if (err != cudaSuccess) 
        {
            std::cerr << "cudaMalloc conv1d device2host d_output failed: " << cudaGetErrorString(err) << std::endl;
            Free_cudapointer(d_output);
            return;
        }

        //cudaMemcpy(d_input, h_input, input_length * in_channels * sizeof(float), cudaMemcpyHostToDevice);

        dim3 numBlocks((out_channels * input_length + 511) / 512);
        dim3 ThreadsPerBlock(512);

        conv1d_kernel <<<numBlocks, ThreadsPerBlock>>> (d_input, d_weights, d_biases, d_output, in_channels, out_channels, input_length, kernel_size);
        //cudaDeviceSynchronize();
        
        cudaMemcpy(h_output, d_output, out_channels * input_length * sizeof(float), cudaMemcpyDeviceToHost);
        
        //Free_cudapointer(d_input);
        Free_cudapointer(d_output);
    }

    void add_DeviceToDevice(int input_length, float* d_input, float* d_add_source, float* d_output)
    {
        //cudaMalloc(&d_input, input_length * 3 * sizeof(float));
        //cudaMalloc(&d_output, input_length * 3 * sizeof(float));
        //cudaMemcpy(d_input, h_input, input_length * 3 * sizeof(float), cudaMemcpyHostToDevice);

        dim3 numBlocks((3 * input_length + 511) / 512);
        dim3 ThreadsPerBlock(512);        
        add_kernel <<<numBlocks, ThreadsPerBlock>>> (input_length, d_input, d_add_source, d_output);

        //cudaMemcpy(h_output, d_output, input_length * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    }

    void transpose2_HostToDevice(int input_length, float* h_input, float* d_output, float* d_output2)
    {
        cudaMalloc(&d_input, input_length * 3 * sizeof(float));
        //cudaMalloc(&d_output, input_length * 3 * sizeof(float));
        cudaMemcpy(d_input, h_input, input_length * 3 * sizeof(float), cudaMemcpyHostToDevice);

        dim3 numBlocks((3 * input_length + 511) / 512);
        dim3 ThreadsPerBlock(512);        
        transpose_kernel <<<numBlocks, ThreadsPerBlock>>> (input_length, d_input, d_output, d_output2);

        //cudaMemcpy(h_output, d_output, input_length * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    }


};

__global__ void BatchNorm1D_kernel_relu(const float* d_input, float* d_output, const float* d_weights,
                                    const float* d_biases, const float* d_running_mean, const float* d_running_var,
                                    int input_length, int numFeature)
{
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_col = global_idx % input_length;
    int idx_row = global_idx / input_length;
    // row表示行号，col表示列号
    if(global_idx < input_length * numFeature)
    {
        float value = (d_input[global_idx] - d_running_mean[idx_row]) / sqrtf(d_running_var[idx_row] + 1e-5);
        d_output[global_idx] = max(d_weights[idx_row] * value + d_biases[idx_row], 0.0); 
        // 内执行relu
    }
}

__global__ void BatchNorm1D_kernel_norelu(const float* d_input, float* d_output, const float* d_weights,
                                    const float* d_biases, const float* d_running_mean, const float* d_running_var,
                                    int input_length, int numFeature)
{
    int global_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_col = global_idx % input_length;
    int idx_row = global_idx / input_length;
    // row表示行号，col表示列号
    if(global_idx < input_length * numFeature)
    {
        float value = (d_input[global_idx] - d_running_mean[idx_row]) / sqrtf(d_running_var[idx_row] + 1e-5);
        d_output[global_idx] = d_weights[idx_row] * value + d_biases[idx_row]; 
        // 内执行relu
    }
}

class BatchNorm1D
{
public:
    int numFeature, input_length;  // 成员变量修复
    std::vector<float>weights, biases, num_batches_tracked, running_mean, running_var;
    float *h_input;
    float *d_input, *d_output, *d_weights, *d_biases, *d_running_mean, *d_running_var; // 修复指针声明

    BatchNorm1D(int numFeature, std::vector<float>weights, std::vector<float>biases, std::vector<float>running_mean, std::vector<float>running_var)
        : numFeature(numFeature), weights(weights), biases(biases), running_mean(running_mean), running_var(running_var)
    {
        cudaError_t err;

        err = cudaMalloc(&d_weights, numFeature * sizeof(float));
        if (err != cudaSuccess) 
            std::cerr << "cudaMalloc d_weights failed: " << cudaGetErrorString(err) << std::endl;

        err = cudaMalloc(&d_biases, numFeature * sizeof(float));
        if (err != cudaSuccess) 
            std::cerr << "cudaMalloc d_biases failed: " << cudaGetErrorString(err) << std::endl;

        err = cudaMalloc(&d_running_mean, numFeature * sizeof(float));
        if (err != cudaSuccess) 
            std::cerr << "cudaMalloc d_running_mean failed: " << cudaGetErrorString(err) << std::endl;

        err = cudaMalloc(&d_running_var, numFeature * sizeof(float));
        if (err != cudaSuccess) 
            std::cerr << "cudaMalloc d_running_var failed: " << cudaGetErrorString(err) << std::endl;


        cudaMemcpy(d_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_biases, biases.data(), biases.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_running_mean, running_mean.data(), running_mean.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_running_var, running_var.data(), running_var.size()*sizeof(float), cudaMemcpyHostToDevice);

    }
    // 析构函数：释放 GPU 内存
    ~BatchNorm1D() 
    {
        // vector 有自动内存回收机制不需要析构

        cudaFree(d_weights);
        cudaFree(d_biases);
        cudaFree(d_running_mean);
        cudaFree(d_running_var);
    }
    // 前向传播函数：调用 CUDA 内核函数
    void forward(int input_length, float* h_input, float* h_output) 
    {   
        cudaError_t err;
        err = cudaMalloc(&d_input, input_length * numFeature * sizeof(float));
        if (err != cudaSuccess) 
            std::cerr << "cudaMalloc d_input failed: " << cudaGetErrorString(err) << std::endl;

        err = cudaMalloc(&d_output, input_length * numFeature * sizeof(float));
        if (err != cudaSuccess) 
            std::cerr << "cudaMalloc d_output failed: " << cudaGetErrorString(err) << std::endl;

        cudaMemcpy(d_input, h_input, input_length * numFeature * sizeof(float), cudaMemcpyHostToDevice);
        dim3 numBlocks((numFeature * input_length + 511) / 512);
        dim3 ThreadsPerBlock(512);

        //int blockSize = 512;
        //int gridSize  = out_channels * input_length / blockSize;
        BatchNorm1D_kernel_relu<<<numBlocks, ThreadsPerBlock>>>(d_input, d_output, d_weights, d_biases, d_running_mean, d_running_var, 
                                                            input_length, numFeature);
        //cudaDeviceSynchronize();
        
        cudaMemcpy(h_output, d_output, numFeature * input_length * sizeof(float), cudaMemcpyDeviceToHost);
        //return h_output;

        if (d_input != nullptr) 
        {
            cudaFree(d_input);
            d_input = nullptr;  // 避免悬空指针
        }
        if (d_output != nullptr) 
        {
            cudaFree(d_output);
            d_output = nullptr;  // 避免悬空指针
        }
    }

    void forward_DeviceToDevice(int input_length, float* d_input, float* d_output) 
    {   
        cudaError_t err;

        dim3 numBlocks((numFeature * input_length + 511) / 512);
        dim3 ThreadsPerBlock(512);

        //int blockSize = 512;
        //int gridSize  = out_channels * input_length / blockSize;
        BatchNorm1D_kernel_relu<<<numBlocks, ThreadsPerBlock>>>(d_input, d_output, d_weights, d_biases, d_running_mean, d_running_var, 
                                                            input_length, numFeature);
        //cudaDeviceSynchronize();
        
        //cudaMemcpy(h_output, d_output, numFeature * input_length * sizeof(float), cudaMemcpyDeviceToHost);
        //return h_output;
    }


    // 前向传播函数：调用 CUDA 内核函数
    void forward_norelu_DeviceToDevice(int input_length, float* d_input, float* d_output) 
    {   
        cudaError_t err;

        cudaMemcpy(d_input, h_input, input_length * numFeature * sizeof(float), cudaMemcpyHostToDevice);
        dim3 numBlocks((numFeature * input_length + 511) / 512);
        dim3 ThreadsPerBlock(512);

        //int blockSize = 512;
        //int gridSize  = out_channels * input_length / blockSize;
        BatchNorm1D_kernel_norelu<<<numBlocks, ThreadsPerBlock>>>(d_input, d_output, d_weights, d_biases, d_running_mean, d_running_var, 
                                                            input_length, numFeature);
        //cudaDeviceSynchronize();
        
        //cudaMemcpy(h_output, d_output, numFeature * input_length * sizeof(float), cudaMemcpyDeviceToHost);
        //return h_output;
    }


};

__global__ void torch_max0(int input_length, int numFeature, float* d_input, float* d_output)
{
    //float *d_input, *d_output;
    //cudaMalloc(&d_input,  input_length * numFeature * sizeof(float));
    //cudaMalloc(&d_output, numFeature * sizeof(float));

    //cudaMemcpy(d_input, h_input, input_length * numFeature * sizeof(float), cudaMemcpyHostToDevice);
    //int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_col =(blockDim.x * blockIdx.x + threadIdx.x) % input_length;
    int idx_row =(blockDim.x * blockIdx.x + threadIdx.x) / input_length; 
    // matx 是x方向坐标，即一组数据， maty是y方向坐标，即不同的特征；
    if(idx_row < numFeature && idx_col == 0)
    {
        float maxVal = -1;
        for(int j = 0; j < input_length; ++j)
            maxVal = fmaxf(maxVal, d_input[idx_row * input_length + j]);

        d_output[idx_row] = maxVal;
    }
    //cudaMemcpy(h_output, d_output, numFeature * sizeof(float), cudaMemcpyDeviceToHost);
};

__global__ void torch_bmm0(int input_length, int channels, float* d_input, float* d_trans, float* d_output)
{
    /*
        用于实现torh.bmm函数以及其上下的两个转置。
    */

    int idx_col =(blockDim.x * blockIdx.x + threadIdx.x) % input_length;
    int idx_row =(blockDim.x * blockIdx.x + threadIdx.x) / input_length; 
    // matx 是x方向坐标，即一组数据， maty是y方向坐标，即不同的特征；
    if(idx_row < channels && idx_col < input_length)
    {
        float sum = 0;
        for(int i=0; i<channels; ++i)
            sum += d_input[idx_col + i * input_length] * d_trans[i * channels + idx_row];
        d_output[idx_col + idx_row * input_length] = sum;
    }
    //cudaMemcpy(h_output, d_output, numFeature * sizeof(float), cudaMemcpyDeviceToHost);
};


int main(int argc, char *argv[])
{
    //std::cout <<"program start\n";
    std::string dir = argv[1];
   
    // 读取测试集数据
    std::string file_path = "./data/test_point_clouds.h5";
    std::vector<std::vector<float>> list_of_points;
    std::vector<int> list_of_labels;

    //std::cout <<"start  loading data\n";
    read_h5_file(file_path, list_of_points, list_of_labels);
    //std::cout <<"finish loading data\n";    
//    std::cout << list_of_labels.size() <<" " << list_of_points.size()<<" ";
//    std::cout << list_of_points[0].size()<<std::endl;

    //for(int i=0;i<=10;i++)
    //    std::cout << list_of_points[i].size() << std::endl;

    // 读取.txt中的权重
    //std::vector<<read_params("./weights");
    //std::cout <<"start  loading weights\n";
    
    std::map<std::string, std::vector<float>>params_all = read_params(dir);
    //std::cout <<"finish loading data\n";        
    //std::cout << params_all["bn1.bias"];
    // 遍历并打印所有参数
    //for (const auto& pair : params_all) {
    //    std::cout << "Parameter Name: " << pair.first << "\n";
        //std::cout << "\n";
    //}

    //std::cout << params_all["bn1.bias"].size();



// STN 3d
    Conv1D STD3d_conv1(3, 64, 1, params_all["feat.stn.conv1.weight"], params_all["feat.stn.conv1.bias"]);
    Conv1D STD3d_conv2(64, 128, 1, params_all["feat.stn.conv2.weight"], params_all["feat.stn.conv2.bias"]);
    Conv1D STD3d_conv3(128, 1024, 1, params_all["feat.stn.conv3.weight"], params_all["feat.stn.conv3.bias"]);
    Conv1D STD3d_fc1(1024, 512, 1, params_all["feat.stn.fc1.weight"], params_all["feat.stn.fc1.bias"]);
    Conv1D STD3d_fc2(512, 256, 1, params_all["feat.stn.fc2.weight"], params_all["feat.stn.fc2.bias"]);
    Conv1D STD3d_fc3(256, 9, 1, params_all["feat.stn.fc3.weight"], params_all["feat.stn.fc3.bias"]);
    //Conv1D STD3d_fc4(256, 10, 1, params_all["feat.stn.fc4.weight"], params_all["feat.stn.fc4.bias"]);    

    BatchNorm1D STD3d_bn1(64, params_all["feat.stn.bn1.weight"], params_all["feat.stn.bn1.bias"], 
                        params_all["feat.stn.bn1.running_mean"], params_all["feat.stn.bn1.running_var"]);
    BatchNorm1D STD3d_bn2(128, params_all["feat.stn.bn2.weight"], params_all["feat.stn.bn2.bias"], 
                        params_all["feat.stn.bn2.running_mean"], params_all["feat.stn.bn2.running_var"]);
    BatchNorm1D STD3d_bn3(1024, params_all["feat.stn.bn3.weight"], params_all["feat.stn.bn3.bias"], 
                        params_all["feat.stn.bn3.running_mean"], params_all["feat.stn.bn3.running_var"]);
    BatchNorm1D STD3d_bn4(512, params_all["feat.stn.bn4.weight"], params_all["feat.stn.bn4.bias"], 
                        params_all["feat.stn.bn4.running_mean"], params_all["feat.stn.bn4.running_var"]);
    BatchNorm1D STD3d_bn5(256, params_all["feat.stn.bn5.weight"], params_all["feat.stn.bn5.bias"], 
                        params_all["feat.stn.bn5.running_mean"], params_all["feat.stn.bn5.running_var"]);
// PointNetEncoder
    Conv1D PointNetEncoder_conv1(3, 64, 1, params_all["feat.conv1.weight"], params_all["feat.conv1.bias"]);
    Conv1D PointNetEncoder_conv2(64, 128, 1, params_all["feat.conv2.weight"], params_all["feat.conv2.bias"]);
    Conv1D PointNetEncoder_conv3(128, 1024, 1, params_all["feat.conv3.weight"], params_all["feat.conv3.bias"]);
  
    BatchNorm1D PointNetEncoder_bn1(64, params_all["feat.bn1.weight"], params_all["feat.bn1.bias"], 
                        params_all["feat.bn1.running_mean"], params_all["feat.bn1.running_var"]);
    BatchNorm1D PointNetEncoder_bn2(128, params_all["feat.bn2.weight"], params_all["feat.bn2.bias"], 
                        params_all["feat.bn2.running_mean"], params_all["feat.bn2.running_var"]);
    BatchNorm1D PointNetEncoder_bn3(1024, params_all["feat.bn3.weight"], params_all["feat.bn3.bias"], 
                        params_all["feat.bn3.running_mean"], params_all["feat.bn3.running_var"]);


// STN Kd
    Conv1D STDkd_conv1(64, 64, 1, params_all["feat.fstn.conv1.weight"], params_all["feat.fstn.conv1.bias"]);
    Conv1D STDkd_conv2(64, 128, 1, params_all["feat.fstn.conv2.weight"], params_all["feat.fstn.conv2.bias"]);
    Conv1D STDkd_conv3(128, 1024, 1, params_all["feat.fstn.conv3.weight"], params_all["feat.fstn.conv3.bias"]);
    Conv1D STDkd_fc1(1024, 512, 1, params_all["feat.fstn.fc1.weight"], params_all["feat.fstn.fc1.bias"]);
    Conv1D STDkd_fc2(512, 256, 1, params_all["feat.fstn.fc2.weight"], params_all["feat.fstn.fc2.bias"]);
    Conv1D STDkd_fc3(256, 64*64, 1, params_all["feat.fstn.fc3.weight"], params_all["feat.fstn.fc3.bias"]);
    //Conv1D STDkd_fc4(256, 10, 1, params_all["feat.fstn.fc4.weight"], params_all["feat.fstn.fc4.bias"]);


    BatchNorm1D STDkd_bn1(64, params_all["feat.fstn.bn1.weight"], params_all["feat.fstn.bn1.bias"], 
                            params_all["feat.fstn.bn1.running_mean"], params_all["feat.fstn.bn1.running_var"]);
    BatchNorm1D STDkd_bn2(128, params_all["feat.fstn.bn2.weight"], params_all["feat.fstn.bn2.bias"], 
                            params_all["feat.fstn.bn2.running_mean"], params_all["feat.fstn.bn2.running_var"]);
    BatchNorm1D STDkd_bn3(1024, params_all["feat.fstn.bn3.weight"], params_all["feat.fstn.bn3.bias"], 
                            params_all["feat.fstn.bn3.running_mean"], params_all["feat.fstn.bn3.running_var"]);
    BatchNorm1D STDkd_bn4(512, params_all["feat.fstn.bn4.weight"], params_all["feat.fstn.bn4.bias"], 
                            params_all["feat.fstn.bn4.running_mean"], params_all["feat.fstn.bn4.running_var"]);
    BatchNorm1D STDkd_bn5(256, params_all["feat.fstn.bn5.weight"], params_all["feat.fstn.bn5.bias"], 
                            params_all["feat.fstn.bn5.running_mean"], params_all["feat.fstn.bn5.running_var"]);    

// get_model function

    Conv1D get_model_conv1(1024, 512, 1, params_all["fc1.weight"], params_all["fc1.bias"]);
    Conv1D get_model_conv2(512, 256, 1, params_all["fc2.weight"], params_all["fc2.bias"]);
    Conv1D get_model_conv3(256, 10, 1, params_all["fc3.weight"], params_all["fc3.bias"]);

    BatchNorm1D get_model_bn1(512, params_all["bn1.weight"], params_all["bn1.bias"], 
                                params_all["bn1.running_mean"], params_all["bn1.running_var"]);
    BatchNorm1D get_model_bn2(256, params_all["bn2.weight"], params_all["bn2.bias"], 
                                params_all["bn2.running_mean"], params_all["bn2.running_var"]);


    int total_score = 0;
    int MAX_INPUT_LENGTH = 34800;
    double memcpy_time = 0;

    //std::cout << list_of_points[0].size();

    float *x_transposed = (float*) malloc(MAX_INPUT_LENGTH * 3 * sizeof(float));

    float *x_3 = (float*) malloc(MAX_INPUT_LENGTH * 3 * sizeof(float));
    float *x_64 = (float*) malloc(MAX_INPUT_LENGTH * 64 * sizeof(float));
    float *x_128 = (float*) malloc(MAX_INPUT_LENGTH * 128 * sizeof(float));
    float *x_1024 = (float*) malloc(MAX_INPUT_LENGTH * 1024 * sizeof(float));

    float *h_xsingle_10 = (float*) malloc( 10 * sizeof(float));
    float *h_xsingle_256 = (float*) malloc(64 * sizeof(float));
    float *h_xsingle_512 = (float*) malloc(128 * sizeof(float));
    float *h_xsingle_1024 = (float*) malloc(1024 * sizeof(float));

    //float *h_xsingle_10;

    float* d_x_transposed, *d_x_transposed_ret, *d_x_3, *d_x_64, *d_x_128, *d_x_1024, *d_xsingle_10, *d_xsingle_256, *d_xsingle_512, *d_xsingle_1024;  
    float* d_trans, *d_trans_feat, *d_trans_added, *d_trans_feat_added, *d_x_stdkn_64;

    cudaMalloc(&d_x_transposed, MAX_INPUT_LENGTH * 3 * sizeof(float));
    cudaMalloc(&d_x_transposed_ret, MAX_INPUT_LENGTH * 3 * sizeof(float));

    cudaMalloc(&d_x_3, MAX_INPUT_LENGTH * 3 * sizeof(float));    
    cudaMalloc(&d_x_64, MAX_INPUT_LENGTH * 64 * sizeof(float));
    cudaMalloc(&d_x_128, MAX_INPUT_LENGTH * 128 * sizeof(float));
    cudaMalloc(&d_x_1024, MAX_INPUT_LENGTH * 1024 * sizeof(float));
    cudaMalloc(&d_xsingle_10,  10 * sizeof(float));
    cudaMalloc(&d_xsingle_256,  256 * sizeof(float));
    cudaMalloc(&d_xsingle_512,  512 * sizeof(float));
    cudaMalloc(&d_xsingle_1024, 1024 * sizeof(float));   

    cudaMalloc(&d_trans, 9 * sizeof(float));   
    cudaMalloc(&d_trans_added, 9 * sizeof(float));
    float h_trans_added[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    cudaMemcpy(d_trans_added, h_trans_added, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&d_x_stdkn_64, MAX_INPUT_LENGTH * 64 * sizeof(float));

    cudaMalloc(&d_trans_feat, 64 * 64 * sizeof(float));

    cudaMalloc(&d_trans_feat_added, 64 * 64 * sizeof(float));
    float h_trans_feat_added[64 * 64] = {0.0f};
    for(int i=0; i<64; ++i)
        h_trans_feat_added[i * (64 + 1)] += 1.0f;
    
    cudaMemcpy(d_trans_feat_added, h_trans_feat_added, 64*64*sizeof(float), cudaMemcpyHostToDevice);
    //float *x_origin = list_of_points[0].data();

    float *d_STD3d_residual, *d_STDkd_residual;
    cudaMalloc(&d_STD3d_residual, 10 * sizeof(float));    
    cudaMalloc(&d_STDkd_residual, 10 * sizeof(float));


    auto start_time = std::chrono::steady_clock::now();


    for(int k=0; k<list_of_labels.size(); ++k)
    {
    // main branch
        cudaError_t  err;
        
        float *x_origin = list_of_points[k].data();
        int input_length =  int(list_of_points[k].size())/3;  // 输入长度/3 
        //std::cout  << input_length << std::endl;

        STD3d_conv1.transpose2_HostToDevice(input_length, x_origin, d_x_3, d_x_transposed);
        //STD3d_conv1.transpose_HostToDevice(input_length, x_origin, d_x_transposed);
        //cudaMemcpy(d_x_transposed, d_x_3, input_length * 3 * sizeof(float), cudaMemcpyDeviceToDevice);

        STD3d_conv1.forward_DeviceToDevice(input_length, d_x_3, d_x_64);
        STD3d_bn1.forward_DeviceToDevice(input_length, d_x_64, d_x_64);

        //for(int i=0; i<0 + 10; ++i)
        //    std::cout <<x_64[i] << " ";

        STD3d_conv2.forward_DeviceToDevice(input_length, d_x_64, d_x_128);
        STD3d_bn2.forward_DeviceToDevice(input_length, d_x_128, d_x_128);
        STD3d_conv3.forward_DeviceToDevice(input_length, d_x_128, d_x_1024);
        STD3d_bn3.forward_norelu_DeviceToDevice(input_length, d_x_1024, d_x_1024);


        dim3 torch_max0_numBlocks((input_length * 1024 + 511) / 512);
        dim3 torch_max0_ThreadsPerBlock(512);
        //float *d_input, *d_output;
        //cudaMalloc(&d_input, input_length * 1024 * sizeof(float));
        //cudaMalloc(&d_output, 1024 * sizeof(float));    

        //cudaMemcpy(d_input, x_1024, input_length * 1024 * sizeof(float), cudaMemcpyHostToDevice);

        torch_max0<<<torch_max0_numBlocks, torch_max0_ThreadsPerBlock>>>(input_length, 1024, d_x_1024, d_xsingle_1024); 
        //cudaDeviceSynchronize();
        //cudaMemcpy(xsingle_1024, d_output, 1024 * sizeof(float), cudaMemcpyDeviceToHost);

        //Free_cudapointer(d_input);
        //Free_cudapointer(d_output);
        //d_xsingle_1024 正确

        // 1024 是numFeature,固定数字
        STD3d_fc1.forward_DeviceToDevice(1, d_xsingle_1024, d_xsingle_512);
        STD3d_bn4.forward_DeviceToDevice(1, d_xsingle_512, d_xsingle_512);
        STD3d_fc2.forward_DeviceToDevice(1, d_xsingle_512, d_xsingle_256);
        STD3d_bn5.forward_DeviceToDevice(1, d_xsingle_256, d_xsingle_256);
        
        //float *trans = (float*) malloc(9 * sizeof(float));
        
        
        STD3d_fc3.forward_DeviceToDevice(1, d_xsingle_256, d_trans);

        STD3d_fc3.add_DeviceToDevice(9, d_trans, d_trans_added, d_trans);

        // d_trans 正确
        // 残差
        //STD3d_fc4.forward_DeviceToDevice(1, d_xsingle_256, d_STD3d_residual);




        //trans[0] += 1.0f;
        //trans[4] += 1.0f;
        //trans[8] += 1.0f;//iden

        //for(int i=0; i<9; ++i)
        //    std::cout<<trans[i] << " ";
        
        // 正确
        //float *d_trans;
        //cudaMalloc(&d_input, input_length * 3 * sizeof(float));
        //cudaMalloc(&d_output, input_length * 3 * sizeof(float));
        //cudaMalloc(&d_trans, 3 * 3 * sizeof(float));    

        //cudaMemcpy(d_input, x_transposed, input_length * 3 * sizeof(float), cudaMemcpyHostToDevice);
        //cudaMemcpy(d_trans, trans, 9 * sizeof(float), cudaMemcpyHostToDevice);

        dim3 torch_bmm0_numBlocks((input_length * 3 + 511) / 512);
        dim3 torch_bmm0_ThreadsPerBlock(512);

        torch_bmm0<<<torch_bmm0_numBlocks, torch_bmm0_ThreadsPerBlock>>>(input_length, 3, d_x_transposed, d_trans, d_x_transposed_ret);
        //cudaDeviceSynchronize();
        //cudaMemcpy(x_transposed, d_output, input_length * 3 * sizeof(float), cudaMemcpyDeviceToHost);

        //Free_cudapointer(d_input);
        //Free_cudapointer(d_output);
        //Free_cudapointer(d_trans);

        //for(int i=19000; i<19000+ 9; ++i)
        //    std::cout<<x_transposed[i] << " ";    
        // d_x_transposed_ret 正确

        PointNetEncoder_conv1.forward_DeviceToDevice(input_length, d_x_transposed_ret, d_x_64);
        PointNetEncoder_bn1.forward_DeviceToDevice(input_length, d_x_64, d_x_64);

        // x_64 正确


        //float* x_stdkn_64 = (float*) malloc(input_length * 64 * sizeof(float));
        //cudaMemcpy(d_x_stdkn_64, d_x_64, input_length * 64 * sizeof(float), cudaMemcpyDeviceToDevice);

        //此处还可优化
        STDkd_conv1.forward_DeviceToDevice(input_length, d_x_64, d_x_stdkn_64);
        STDkd_bn1.forward_DeviceToDevice(input_length, d_x_stdkn_64, d_x_stdkn_64);
        STDkd_conv2.forward_DeviceToDevice(input_length, d_x_stdkn_64, d_x_128);
        STDkd_bn2.forward_DeviceToDevice(input_length, d_x_128, d_x_128);
        STDkd_conv3.forward_DeviceToDevice(input_length, d_x_128, d_x_1024);
        STDkd_bn3.forward_DeviceToDevice(input_length, d_x_1024, d_x_1024);

        // x_1024 正确
    
        dim3 STDkd_torch_max0_numBlocks((input_length * 1024 + 511) / 512);
        dim3 STDkd_torch_max0_ThreadsPerBlock(512);

       // cudaMalloc(&d_input, input_length * 1024 * sizeof(float));
        //cudaMalloc(&d_output, 1024 * sizeof(float));    
        //cudaMemcpy(d_input, x_1024, input_length * 1024 * sizeof(float), cudaMemcpyHostToDevice);

        torch_max0<<<STDkd_torch_max0_numBlocks, STDkd_torch_max0_ThreadsPerBlock>>>(input_length, 1024, d_x_1024, d_xsingle_1024); 
        //cudaDeviceSynchronize();
        //cudaMemcpy(xsingle_1024, d_output, 1024 * sizeof(float), cudaMemcpyDeviceToHost);


        //d_xsingle_1024 正确 sure


        //Free_cudapointer(d_input);
        //Free_cudapointer(d_output);

        STDkd_fc1.forward_DeviceToDevice(1, d_xsingle_1024, d_xsingle_512);
        STDkd_bn4.forward_DeviceToDevice(1, d_xsingle_512, d_xsingle_512);        
        STDkd_fc2.forward_DeviceToDevice(1, d_xsingle_512, d_xsingle_256);
        STDkd_bn5.forward_DeviceToDevice(1, d_xsingle_256, d_xsingle_256);   

        //float *trans_feat = (float*) malloc(64 * 64 * sizeof(float));
        STDkd_fc3.forward_DeviceToDevice(1, d_xsingle_256, d_trans_feat);


        STDkd_fc3.add_DeviceToDevice(64*64, d_trans_feat, d_trans_feat_added, d_trans_feat);
        //for(int i=0; i<64; ++i)
        //    trans_feat[i*(64+1)] += 1.0f;
        //STDkd_fc4.forward_DeviceToDevice(1, d_xsingle_256, d_STDkd_residual);


        //d_trans_feat正确
        //float *d_trans_feat;
        dim3 STDkd_torch_bmm0_numBlocks((input_length * 64 + 511) / 512);
        dim3 STDkd_torch_bmm0_ThreadsPerBlock(512);


        // d_input 分配内存
        //cudaError_t err = cudaMalloc(&d_input, input_length * 64 * sizeof(float));
        //if (err != cudaSuccess) {
        //    std::cerr << "cudaMalloc for d_input failed: " << cudaGetErrorString(err) << std::endl;
        //    return;
        //}
        // d_output 分配内存
        //err = cudaMalloc(&d_output, input_length * 64 * sizeof(float));
        //if (err != cudaSuccess) {
        //    std::cerr << "cudaMalloc for d_output failed: " << cudaGetErrorString(err) << std::endl;
        //    cudaFree(d_input); // 清理已经分配的内存
        //    return;
        //}
        // d_trans 分配内存
        //err = cudaMalloc(&d_trans_feat, 64 * 64 * sizeof(float));
        //if (err != cudaSuccess) {
        //    std::cerr << "cudaMalloc for d_trans failed: " << cudaGetErrorString(err) << std::endl;
        //    cudaFree(d_input); 
        //    cudaFree(d_output);
        //    return;
        //}

        // 确保成功后再进行cudaMemcpy操作
    
        //cudaMemcpy(d_input, x_stdkn_64, input_length * 64 * sizeof(float), cudaMemcpyHostToDevice);

        //cudaMemcpy(d_trans_feat, trans_feat, 64 * 64 * sizeof(float), cudaMemcpyHostToDevice);




        torch_bmm0<<<STDkd_torch_bmm0_numBlocks, STDkd_torch_bmm0_ThreadsPerBlock>>>(input_length, 64, d_x_64, d_trans_feat, d_x_stdkn_64);
        //cudaDeviceSynchronize();

        //cudaMemcpy(x_64, d_output, input_length * 64 * sizeof(float), cudaMemcpyDeviceToHost);

        
        //Free_cudapointer(d_input);
        //Free_cudapointer(d_output);
        //Free_cudapointer(d_trans_feat);

        PointNetEncoder_conv2.forward_DeviceToDevice(input_length, d_x_stdkn_64, d_x_128); 
        PointNetEncoder_bn2.forward_DeviceToDevice(input_length, d_x_128, d_x_128); 
        PointNetEncoder_conv3.forward_DeviceToDevice(input_length, d_x_128, d_x_1024); 
        PointNetEncoder_bn3.forward_norelu_DeviceToDevice(input_length, d_x_1024, d_x_1024);  

        dim3 PointNet_torch_max0_numBlocks((input_length * 1024 + 511) / 512);
        dim3 PointNet_torch_max0_ThreadsPerBlock(512);

        //cudaMalloc(&d_input, input_length * 1024 * sizeof(float));
        //cudaMalloc(&d_output, 1024 * sizeof(float));    
        //cudaMemcpy(d_input, x_1024, input_length * 1024 * sizeof(float), cudaMemcpyHostToDevice);

        torch_max0<<<PointNet_torch_max0_numBlocks, PointNet_torch_max0_ThreadsPerBlock>>>(input_length, 1024, d_x_1024, d_xsingle_1024); 
        //cudaDeviceSynchronize();
        //cudaMemcpy(xsingle_1024, d_output, 1024 * sizeof(float), cudaMemcpyDeviceToHost);
    
        //Free_cudapointer(d_input);
        //Free_cudapointer(d_output);
        
        //for(int i=0; i<0+ 100; ++i)
        //    std::cout<<xsingle_1024[i] << " ";    


        get_model_conv1.forward_DeviceToDevice(1, d_xsingle_1024, d_xsingle_512); 
        get_model_bn1.forward_DeviceToDevice(1, d_xsingle_512, d_xsingle_512); 
        get_model_conv2.forward_DeviceToDevice(1, d_xsingle_512, d_xsingle_256);   
        get_model_bn2.forward_DeviceToDevice(1, d_xsingle_256, d_xsingle_256);
        //std::cout <<" error found !\n";
        get_model_conv3.forward_DeviceToDevice(1, d_xsingle_256, d_xsingle_10);     

        //for(int i=0; i<10; ++i)
        //    std::cout<<xsingle_10[i] << " ";    
        //STDkd_fc3.add_DeviceToDevice(10, d_xsingle_10, d_STD3d_residual, d_xsingle_10);
        //STDkd_fc3.add_DeviceToDevice(10, d_xsingle_10, d_STDkd_residual, d_xsingle_10);        

        cudaMemcpy(h_xsingle_10, d_xsingle_10, 10 * sizeof(float), cudaMemcpyDeviceToHost);

        int sum = 0;
        int maxidx = 0;
        //std::cout <<"\n";

        for(int i=0; i<10; ++i)
        {
            h_xsingle_10[i] = exp(h_xsingle_10[i]);
            //std::cout <<h_xsingle_10[i] <<" ";
            //sum += xsingle_10[i];     

            if(h_xsingle_10[i] > h_xsingle_10[maxidx])
                maxidx = i;
        }

        //std::cout << maxidx <<"  "<< list_of_labels[k] <<"\n";
        if (maxidx == list_of_labels[k])
            total_score ++;

    }
    
    auto end_time = std::chrono::steady_clock::now();

    double query_second = std::chrono::duration<double>(end_time - start_time).count();

    //std::cout <<"\n"<< query_second<< " seconds" << memcpy_time;
    //std::cout <<"\n" << total_score << " "<< list_of_labels.size() << "\n";
    std::cout << std::fixed << std::setprecision(4) << 60 << ":" << float(total_score)/list_of_labels.size();

    free(x_transposed);
    free(x_3);
    free(x_64);
    free(x_128);
    free(x_1024);
    
    free(h_xsingle_10);
    free(h_xsingle_256);
    free(h_xsingle_512);
    free(h_xsingle_1024);
    Free_cudapointer(d_x_transposed);
    Free_cudapointer(d_x_3);
    Free_cudapointer(d_x_64);
    Free_cudapointer(d_x_128);
    Free_cudapointer(d_x_1024);

    Free_cudapointer(d_xsingle_10);
    Free_cudapointer(d_xsingle_256);
    Free_cudapointer(d_xsingle_512);
    Free_cudapointer(d_xsingle_1024);

    Free_cudapointer(d_x_stdkn_64);
    Free_cudapointer(d_trans);
    Free_cudapointer(d_trans_feat);
    Free_cudapointer(d_trans_added);
    Free_cudapointer(d_trans_feat_added);
    return 0;

}