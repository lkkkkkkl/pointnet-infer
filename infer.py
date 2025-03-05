import os
import h5py
import time
import numpy as np
import torch
import triton
import triton.language as tl
import math

from torch.autograd import Variable



def read_params(dir):
    # 列出所有txt文件
    files = [f for f in os.listdir(dir) if f.endswith('.txt')]
    params = {}
    for fileName in files:
        data = []
        with open(os.path.join(dir, fileName), 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip()
                if line:  # 确保不是空行
                    value = float(line)
                    data.append(value)
        modelName = fileName.replace(".txt", "")
        params[modelName] = data
    return params

def read_h5_file(dataPath, sample_length):
    np.random.seed(2024)
    list_of_points = []
    list_of_labels = []
    with h5py.File(dataPath, "r") as hf:
        for k in hf.keys():
            # 读取 points 数据，形状为 (N, 3)
            points = hf[k]["points"][:].astype(np.float32)  # 确保数据类型为 float32
            N, dim = points.shape
            assert dim == 3, f"Expected 3 dimensions, got {dim} dimensions."
            indices = np.random.choice(N, sample_length, replace=False)
            sampled_points = points[indices, :]  # 采样后的形状为 (sample_length, 3)
            #print(indices)
            reshaped_points = sampled_points.T     # 重塑为 (3, sample_length)
            list_of_points.append(reshaped_points)
            list_of_labels.append(hf[k].attrs["label"])
    return list_of_points, list_of_labels


@triton.jit
def conv1d_kernel(
    data_ptr, weights_ptr, bias_ptr, output_ptr,
    data_length, in_channels, out_channels, batch_num,
    BLOCK_SIZE: tl.constexpr):
    # 获取程序的 grid 索引
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    block_id = tl.program_id(2)
    
    # 计算长度维度的起始索引
    block_start = block_id * BLOCK_SIZE
    
    # 计算线程在 block 内的索引
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 创建一个掩码，防止越界访问
    mask = offsets < data_length
    
    # 初始化累加器
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32) + tl.load(bias_ptr + channel_id)
    
    # 计算输入数据和权重的偏移
    for c in range(in_channels):
        weight_val = tl.load(weights_ptr + channel_id * in_channels + c)
        data_offset = batch_id * in_channels * data_length + c * data_length + offsets
        data_val = tl.load(data_ptr + data_offset, mask=mask, other=0.0)
        acc += weight_val * data_val


    # 计算输出数据的偏移
    idx_out = batch_id * out_channels * data_length + channel_id * data_length + offsets
    
    # 将结果存储到输出张量中
    tl.store(output_ptr + idx_out, acc, mask=mask)

class Conv1D:
    def __init__(self, in_channels, out_channels, weights, bias):
        self.in_channels = in_channels
        self.out_channels = out_channels
        # 将权重和偏置转换为 CUDA 张量，并调整形状
        self.weights = torch.tensor(weights, device='cuda', dtype=torch.float32).view(out_channels, in_channels).contiguous()
        self.bias = torch.tensor(bias, device='cuda', dtype=torch.float32).contiguous()
    
    def forward(self, data):
        """
        前向传播
        Args:
            data (torch.Tensor): 输入张量，形状为 (batch_num, in_channels, input_length)
        Returns:
            torch.Tensor: 输出张量，形状为 (batch_num, out_channels, output_length)
        """
        batch_num, in_channels, input_length = data.shape
        assert in_channels == self.in_channels, print(in_channels, self.in_channels)

    
        # 计算输出长度，kernel_size=1, stride=1
        output_length = input_length  # 因为 kernel_size=1, stride=1
    
        # 初始化输出张量
        output_tensor = torch.empty((batch_num, self.out_channels, output_length), device='cuda', dtype=torch.float32)
    
        # 设置 Triton 网格
        BLOCK_SIZE = 128
        grid = (
            batch_num,
            self.out_channels,
            (input_length + BLOCK_SIZE - 1) // BLOCK_SIZE
        )
    
        # 调用 Triton 内核
        conv1d_kernel[grid](
            data_ptr=data,
            weights_ptr=self.weights,
            bias_ptr=self.bias,
            output_ptr=output_tensor,
            data_length=input_length,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_num=batch_num,
            BLOCK_SIZE=BLOCK_SIZE  # 作为 constexpr 参数传递
        )
    
        return output_tensor


@triton.jit
def bn1d_kernel(
                data_ptr, weights_ptr, bias_ptr, r_mean_ptr, r_var_ptr, output_ptr,
                data_length, in_channels, out_channels, batch_num, 
                BLOCK_SIZE:tl.constexpr):
    
    batch_id   = tl.program_id(0)
    channel_id = tl.program_id(1)
    block_id   = tl.program_id(2)

    block_start = block_id * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < data_length

    weights_val = tl.load(weights_ptr + channel_id)
    bias_val   = tl.load(bias_ptr + channel_id)
    r_mean_val = tl.load(r_mean_ptr + channel_id)
    r_var_val  = tl.load(r_var_ptr + channel_id)

    idx_data = batch_id * in_channels * data_length + channel_id * data_length + offsets
    x = tl.load(data_ptr + idx_data, mask=mask)
    x = (x-r_mean_val)/tl.sqrt(r_var_val + 1e-5)

    y = weights_val * x + bias_val

    y = tl.maximum(y, 0.0)

    idx_output = batch_id * out_channels * data_length + channel_id * data_length + offsets

    tl.store(output_ptr + idx_output, y, mask=mask)


@triton.jit
def bn1d_kernel_norelu(
                data_ptr, weights_ptr, bias_ptr, r_mean_ptr, r_var_ptr, output_ptr,
                data_length, in_channels, out_channels, batch_num, 
                BLOCK_SIZE:tl.constexpr):
    
    batch_id   = tl.program_id(0)
    channel_id = tl.program_id(1)
    block_id   = tl.program_id(2)

    block_start = block_id * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < data_length

    weights_val = tl.load(weights_ptr + channel_id)
    bias_val   = tl.load(bias_ptr + channel_id)
    r_mean_val = tl.load(r_mean_ptr + channel_id)
    r_var_val  = tl.load(r_var_ptr + channel_id)

    idx_data = batch_id * in_channels * data_length + channel_id * data_length + offsets
    x = tl.load(data_ptr + idx_data, mask=mask)
    x = (x-r_mean_val)/tl.sqrt(r_var_val + 1e-5)

    y = weights_val * x + bias_val

    #y = tl.maximum(y, 0.0)

    idx_output = batch_id * out_channels * data_length + channel_id * data_length + offsets

    tl.store(output_ptr + idx_output, y, mask=mask)

class BatchNorm1D:
    def __init__(self, in_channels, weight, bias, running_mean, running_var):
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.weights = torch.tensor(weight, device='cuda', dtype=torch.float32)
        self.bias   = torch.tensor(bias, device='cuda', dtype=torch.float32)
        self.r_mean = torch.tensor(running_mean, device='cuda', dtype=torch.float32)
        self.r_var  = torch.tensor(running_var, device='cuda', dtype=torch.float32)

    def forward(self, data, if_relu=True):
        batch_num, in_channels, input_length = data.shape
        assert in_channels == self.in_channels

        output_length = input_length
        output_tensor = torch.empty((batch_num, self.out_channels, output_length), device='cuda', dtype=torch.float32)

        # 设置 Triton 网格
        BLOCK_SIZE = 128
        grid = (
            batch_num,
            self.out_channels,
            (input_length + BLOCK_SIZE - 1) // BLOCK_SIZE
        )
        if if_relu:
            # 调用 Triton 内核s
            bn1d_kernel[grid](
                data_ptr=data,
                weights_ptr=self.weights,
                bias_ptr=self.bias,
                r_mean_ptr=self.r_mean,
                r_var_ptr=self.r_var,
                output_ptr=output_tensor,
                data_length=input_length,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_num=batch_num,
                BLOCK_SIZE=BLOCK_SIZE  # 作为 constexpr 参数传递
            )
        else:
            bn1d_kernel_norelu[grid](
                data_ptr=data,
                weights_ptr=self.weights,
                bias_ptr=self.bias,
                r_mean_ptr=self.r_mean,
                r_var_ptr=self.r_var,
                output_ptr=output_tensor,
                data_length=input_length,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_num=batch_num,
                BLOCK_SIZE=BLOCK_SIZE  # 作为 constexpr 参数传递
            )            
        return output_tensor

@triton.jit
def max_reduce_kernel(
    x_ptr,        # 输入张量指针
    output_ptr,   # 输出张量指针
    channels: tl.constexpr,
    length: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    
    # 计算偏移量
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # 计算输入数据的起始指针
    x_start_ptr = x_ptr + batch_id * channels * length + channel_id * length
    
    # 初始化最大值为标量
    max_val = -float('inf')
    
    # 遍历长度维度，步长为 BLOCK_SIZE
    for i in range(0, length, BLOCK_SIZE):
        # 当前块的偏移
        current_offsets = offsets + i
        # 创建掩码，防止越界访问
        mask = current_offsets < length
        # 计算当前块的指针
        x_ptrs = x_start_ptr + current_offsets
        # 加载数据
        x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
        # 在当前块内计算最大值
        current_max = tl.max(x, axis=0)
        # 提取标量值
        current_max = current_max.to(tl.float32)
        # 更新全局最大值
        max_val = tl.maximum(max_val, current_max)
    
    # 将结果存储到输出张量中
    out_idx = batch_id * channels + channel_id
    tl.store(output_ptr + out_idx, max_val)


def max_reduce(x):
    batch_size, channels, length = x.shape
    x = x.contiguous()
    
    # 创建输出张量，形状为 (batch_size, channels)
    output = torch.empty((batch_size, channels), device='cuda', dtype=torch.float32)
    
    # 设置 Triton 网格
    grid = (batch_size, channels)
    
    # 调用 Triton 内核
    BLOCK_SIZE = 1024  # 可以根据长度维度大小调整
    max_reduce_kernel[grid](
        x_ptr=x,
        output_ptr=output,
        channels=channels,
        length=length,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # 调整输出形状，保持维度
    output = output.view(batch_size, channels, 1)
    
    return output


@triton.jit
def add_iden_kernel(
    x_ptr,      # 输入 x 的指针
    iden_ptr,   # 输入 iden 的指针
    output_ptr, # 输出的指针
    N,          # 数据的长度（9）
    BLOCK_SIZE: tl.constexpr
):
    # 计算全局线程的索引
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = pid * N + offsets

    mask = offsets < N  # 防止越界

    # 加载 x 和 iden 的数据
    x = tl.load(x_ptr + idx, mask=mask, other=0.0)
    iden = tl.load(iden_ptr + idx, mask=mask, other=0.0)

    # 执行加法
    y = x + iden

    # 将结果存储到输出张量
    tl.store(output_ptr + idx, y, mask=mask)

def add_iden(x, iden):
    batchsize, N = x.shape
    output = torch.empty_like(x)

    BLOCK_SIZE = 1
    while BLOCK_SIZE < N:
        BLOCK_SIZE *= 2

    # 设置 Triton 网格
    grid = (batchsize,)

    # 调用 Triton 内核
    add_iden_kernel[grid](
        x_ptr=x,
        iden_ptr=iden,
        output_ptr=output,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output


def main():
    dir_path = os.path.dirname(__file__)  # 保存模型参数文件(.txt)的文件夹路径
    
    LENGTH = 1024  # 修改为合适的长度

    # 读取模型参数
    params = read_params('./train')
    
    # 读取训练集数据
    dataPath = "./data/test_point_clouds.h5"
    list_of_points, list_of_labels = read_h5_file(dataPath, sample_length=LENGTH)
    
    # 转换为 (batch_num, in_channels, input_length) 形状的张量
    data_test = torch.tensor(list_of_points, dtype=torch.float32).to('cuda').contiguous()
    label_test = torch.tensor(list_of_labels, dtype=torch.int).to('cuda').contiguous()
    batch_num = 1000
    
    print(data_test.shape, data_test.is_cuda)  # 应输出 torch.Size([batch_num, 3, 64]) True
    
    # 创建 Conv1D 实例，假设 'feat.conv1.weight' 长度为 64 * 3 = 192，'feat.conv1.bias' 长度为 64
    STN3D_conv1 = Conv1D(3, 64, params['feat.stn.conv1.weight'], params['feat.stn.conv1.bias'])
    STN3D_conv2 = Conv1D(64, 128, params['feat.stn.conv2.weight'], params['feat.stn.conv2.bias'])
    STN3D_conv3 = Conv1D(128, 1024, params['feat.stn.conv3.weight'], params['feat.stn.conv3.bias'])

    STN3D_bn1   = BatchNorm1D(64, params['feat.stn.bn1.weight'], params['feat.stn.bn1.bias'], params['feat.stn.bn1.running_mean'], params['feat.stn.bn1.running_var'])
    STN3D_bn2   = BatchNorm1D(128, params['feat.stn.bn2.weight'], params['feat.stn.bn2.bias'], params['feat.stn.bn2.running_mean'], params['feat.stn.bn2.running_var'])
    STN3D_bn3   = BatchNorm1D(1024, params['feat.stn.bn3.weight'], params['feat.stn.bn3.bias'], params['feat.stn.bn3.running_mean'], params['feat.stn.bn3.running_var'])
    STN3D_bn4   = BatchNorm1D(512, params['feat.stn.bn4.weight'], params['feat.stn.bn4.bias'], params['feat.stn.bn4.running_mean'], params['feat.stn.bn4.running_var'])
    STN3D_bn5   = BatchNorm1D(256, params['feat.stn.bn5.weight'], params['feat.stn.bn5.bias'], params['feat.stn.bn5.running_mean'], params['feat.stn.bn5.running_var'])
    
    STN3D_fc1 = Conv1D(1024, 512, params['feat.stn.fc1.weight'], params['feat.stn.fc1.bias'])
    STN3D_fc2 = Conv1D(512, 256, params['feat.stn.fc2.weight'], params['feat.stn.fc2.bias'])
    STN3D_fc3 = Conv1D(256, 9, params['feat.stn.fc3.weight'], params['feat.stn.fc3.bias'])

    # 创建 Conv1D 实例，假设 'feat.conv1.weight' 长度为 64 * 3 = 192，'feat.conv1.bias' 长度为 64
    STNkD_conv1 = Conv1D(64, 64, params['feat.fstn.conv1.weight'], params['feat.fstn.conv1.bias'])
    STNkD_conv2 = Conv1D(64, 128, params['feat.fstn.conv2.weight'], params['feat.fstn.conv2.bias'])
    STNkD_conv3 = Conv1D(128, 1024, params['feat.fstn.conv3.weight'], params['feat.fstn.conv3.bias'])

    STNkD_bn1   = BatchNorm1D(64, params['feat.fstn.bn1.weight'], params['feat.fstn.bn1.bias'], params['feat.fstn.bn1.running_mean'], params['feat.fstn.bn1.running_var'])
    STNkD_bn2   = BatchNorm1D(128, params['feat.fstn.bn2.weight'], params['feat.fstn.bn2.bias'], params['feat.fstn.bn2.running_mean'], params['feat.fstn.bn2.running_var'])
    STNkD_bn3   = BatchNorm1D(1024, params['feat.fstn.bn3.weight'], params['feat.fstn.bn3.bias'], params['feat.fstn.bn3.running_mean'], params['feat.fstn.bn3.running_var'])
    STNkD_bn4   = BatchNorm1D(512, params['feat.fstn.bn4.weight'], params['feat.fstn.bn4.bias'], params['feat.fstn.bn4.running_mean'], params['feat.fstn.bn4.running_var'])
    STNkD_bn5   = BatchNorm1D(256, params['feat.fstn.bn5.weight'], params['feat.fstn.bn5.bias'], params['feat.fstn.bn5.running_mean'], params['feat.fstn.bn5.running_var'])
    
    STNkD_fc1 = Conv1D(1024, 512, params['feat.fstn.fc1.weight'], params['feat.fstn.fc1.bias'])
    STNkD_fc2 = Conv1D(512, 256, params['feat.fstn.fc2.weight'], params['feat.fstn.fc2.bias'])
    STNkD_fc3 = Conv1D(256, 64*64, params['feat.fstn.fc3.weight'], params['feat.fstn.fc3.bias'])


    # PointNetEncoder
    PN_conv1 = Conv1D(3, 64, params['feat.conv1.weight'], params['feat.conv1.bias'])
    PN_conv2 = Conv1D(64, 128, params['feat.conv2.weight'], params['feat.conv2.bias'])
    PN_conv3 = Conv1D(128, 1024, params['feat.conv3.weight'], params['feat.conv3.bias'])

    PN_bn1   = BatchNorm1D(64, params['feat.bn1.weight'], params['feat.bn1.bias'], params['feat.bn1.running_mean'], params['feat.bn1.running_var'])
    PN_bn2   = BatchNorm1D(128, params['feat.bn2.weight'], params['feat.bn2.bias'], params['feat.bn2.running_mean'], params['feat.bn2.running_var'])
    PN_bn3   = BatchNorm1D(1024, params['feat.bn3.weight'], params['feat.bn3.bias'], params['feat.bn3.running_mean'], params['feat.bn3.running_var'])

    fc1 = Conv1D(1024, 512, params['fc1.weight'], params['fc1.bias'])
    fc2 = Conv1D(512, 256, params['fc2.weight'], params['fc2.bias'])
    fc3 = Conv1D(256, 10, params['fc3.weight'], params['fc3.bias'])

    bn1   = BatchNorm1D(512, params['bn1.weight'], params['bn1.bias'], params['bn1.running_mean'], params['bn1.running_var'])
    bn2   = BatchNorm1D(256, params['bn2.weight'], params['bn2.bias'], params['bn2.running_mean'], params['bn2.running_var'])



    torch.set_printoptions(precision=6)

    # 开始计时
    start = time.time()



    # 执行前向传播
    x_stn3d = STN3D_conv1.forward(data_test)

    x_stn3d = STN3D_bn1.forward(x_stn3d)
    x_stn3d = STN3D_conv2.forward(x_stn3d)
    x_stn3d = STN3D_bn2.forward(x_stn3d)

    x_stn3d = STN3D_conv3.forward(x_stn3d)
    x_stn3d = STN3D_bn3.forward(x_stn3d)    
    x_stn3d = max_reduce(x_stn3d)

    #print(output.shape)
    x_stn3d = STN3D_fc1.forward(x_stn3d)
    x_stn3d = STN3D_bn4.forward(x_stn3d)
    x_stn3d = STN3D_fc2.forward(x_stn3d)
    x_stn3d = STN3D_bn5.forward(x_stn3d)
    x_stn3d = STN3D_fc3.forward(x_stn3d)

    #print(x_stn3d)

    iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(batch_num, 1)
    iden_stn3d = iden.to('cuda')
    #print(x_stn3d.shape)

    x_stn3d = x_stn3d.squeeze(-1)
    x_stn3d = add_iden(x_stn3d, iden_stn3d)
    feat = x_stn3d.view(batch_num, 3, 3).contiguous()



    #x = data_test
    x = data_test.transpose(2, 1)
    #print(x.shape, x_stn3d.shape)

    x = torch.bmm(x, feat).contiguous()

    x = x.transpose(2, 1).contiguous()
    #print(x.shape)
    # 此处正确

    #print(x, x.shape)
    #print(x, x.shape)
    x = PN_conv1.forward(x).contiguous()
    #print(x, x.shape)
    x = PN_bn1.forward(x).contiguous()
    #print(x, x.shape)

    x_stnkd = STNkD_conv1.forward(x).contiguous()
    x_stnkd = STNkD_bn1.forward(x_stnkd).contiguous()
    x_stnkd = STNkD_conv2.forward(x_stnkd).contiguous()
    x_stnkd = STNkD_bn2.forward(x_stnkd).contiguous()
    x_stnkd = STNkD_conv3.forward(x_stnkd).contiguous()
    x_stnkd = STNkD_bn3.forward(x_stnkd).contiguous()
    x_stnkd = max_reduce(x_stnkd).contiguous()
    #print(x_stnkd)

    x_stnkd = STNkD_fc1.forward(x_stnkd).contiguous()
    x_stnkd = STNkD_bn4.forward(x_stnkd).contiguous() 
    x_stnkd = STNkD_fc2.forward(x_stnkd).contiguous() 
    x_stnkd = STNkD_bn5.forward(x_stnkd).contiguous() 
    x_stnkd = STNkD_fc3.forward(x_stnkd).contiguous() 



    #print(x_stnkd, x_stnkd.shape)


    iden = torch.from_numpy(np.eye(64).flatten().astype(np.float32)).view(1, 64 * 64).repeat(batch_num, 1)
    iden_stnkd = iden.to('cuda')

    x_stnkd = x_stnkd.squeeze(-1)

    #print(iden_stnkd.shape, x_stnkd.shape)
    x_stnkd = add_iden(x_stnkd, iden_stnkd).view(batch_num, 64, 64).contiguous()

    x = x.transpose(2, 1).contiguous()
    #print(x.shape, x_stnkd.shape)

    x = torch.bmm(x, x_stnkd).transpose(2, 1).contiguous()

    x = PN_conv2.forward(x).contiguous()
    x = PN_bn2.forward(x).contiguous()
    x = PN_conv3.forward(x).contiguous()
    x = PN_bn3.forward(x, False).contiguous()
    x = max_reduce(x).contiguous()
    #x = x.squeeze(-1)
    
    x = fc1.forward(x).contiguous()
    x = bn1.forward(x).contiguous()
    x = fc2.forward(x).contiguous()
    x = bn2.forward(x).contiguous()
    x = fc3.forward(x).contiguous()

    #x = x.view(batch_num, 1, 10).contiguous()

    #x = max_reduce(x).contiguous()
    x = x.squeeze(-1).contiguous()

    x_idx = torch.argmax(x, 1).int()

    print(x_idx, x_idx.shape)
    acc = torch.sum(x_idx == label_test).item()
    print(acc)
    #print(x_stnkd.shape)


    #print(x.shape)

    #x.transpose(2, 1)

    #print(x)



    
    # 打印输出的前 100 个元素

    
    # 打印输出形状
    #print("输出张量形状:", output.shape)  # 应输出 torch.Size([batch_num, 64, 64])
    

    # 假设有一个 do_inference 函数
    # accuracy_rate = do_inference(list_of_points, list_of_labels, params)
    # 结束计时
    end = time.time()
    ms = end - start
    
    # 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
    # print(f"{ms:.4f}:{accuracy_rate:.4f}")
    # 由于 do_inference 未定义，暂时注释掉
    print(f"{ms:.4f}:0.0001")

if __name__ == '__main__':
    main()
