import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import torch.nn.parallel
from torch.autograd import Variable
import numpy as np
import h5py
from tqdm import tqdm
import time
import random
from torch.cuda.amp import GradScaler, autocast


# import provider
num_class = 10
total_epoch = 40
script_dir = os.path.dirname(__file__)  # 获取脚本所在的目录
script_dir = os.path.join(script_dir, 'params')

print(script_dir)

def set_random_seed(seed):
    random.seed(seed)  # 固定 Python 的随机种子
    np.random.seed(seed)  # 固定 NumPy 的随机种子
    torch.manual_seed(seed)  # 固定 PyTorch 的随机种子
    torch.cuda.manual_seed(seed)  # 固定 CUDA 的随机种子
    torch.backends.cudnn.deterministic = True  # 确保CuDNN使用确定性算法
    torch.backends.cudnn.benchmark = False  # 禁止使用CuDNN的自动优化
    
class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)


        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))

        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


# 模型定义
class get_model(nn.Module):
    def __init__(self, k=10, normal_channel=False):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


class PointCloudDataset(Dataset):
    def __init__(self,root, split, enhance=False):
        self.list_of_points = []
        self.list_of_labels = []
        self.root = root
        self.split = split

        # 加载点云数据
        with h5py.File(f"{self.root}/{self.split}_point_clouds.h5", "r") as hf:
            for k in hf.keys():
                points = hf[k]["points"][:].astype(np.float32)

                if points.shape[0] < 18000:
                    # 点云数据不足 2 万时，使用插值补到 2 万
                    num_points_needed = 18000 - points.shape[0]
                    new_points = []
                    for _ in range(num_points_needed):
                        random_idx_1 = np.random.randint(0, points.shape[0])
                        random_idx_2 = np.random.randint(0, points.shape[0])
                        point_1 = points[random_idx_1]
                        point_2 = points[random_idx_2]
                        # 在两个点之间插值生成新点
                        alpha = np.random.rand()
                        new_point = point_1 + alpha * (point_2 - point_1)
                        new_points.append(new_point)
                    new_points = np.array(new_points)
                    points = np.vstack([points, new_points])
                elif points.shape[0] > 18000:
                    # 点云数据超过 2 万时，随机均匀删除到 2 万
                    indices = np.random.choice(points.shape[0], 18000, replace=False)
                    points = points[indices]

                self.list_of_points.append(points)
                self.list_of_labels.append(hf[k].attrs["label"])
     
        if enhance:
            target_labels = [0, 3, 8]
            points_to_augment = []
            labels_to_augment = []
            
            for point, label in zip(self.list_of_points, self.list_of_labels):
                if label in target_labels:
                    points_to_augment.append(point)
                    labels_to_augment.append(label)
                    
            for point, label in zip(points_to_augment, labels_to_augment):
                self.list_of_points.append(point)  # 复制点
                self.list_of_labels.append(label)  # 复制标签
      

    def __len__(self):
        return len(self.list_of_points)

    def __getitem__(self, idx):
        points = self.list_of_points[idx]
        label = self.list_of_labels[idx]
        return points, label

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, num_class=10):
    mean_correct = []
    classifier = model.eval()

    with torch.no_grad():  # 关闭梯度计算以节省内存
        for j, (points, target) in enumerate(loader): # 显示进度条
            points, target = points.cuda(), target.cuda()
            
            points = points.transpose(2, 1)  # 调整维度，符合模型输入需求
            pred, _ = classifier(points)  # 执行前向传播
            pred_choice = pred.max(1)[1]  # 获取预测类别
            
            correct = pred_choice.eq(target.long()).cpu().sum()  # 计算预测准确的数量
            mean_correct.append(correct.item() / float(points.size(0)))  # 计算准确率并添加到列表

    instance_acc = np.mean(mean_correct)  # 计算所有批次的平均准确率

    return instance_acc

def pad_collate_fn(batch):
    # 找到批次中最小的数组大小
    min_size = min([item[0].shape[0] for item in batch])
    
    # 截断数组
    padded_batch = []
    for points, target in batch:
        # 截断数组
        points = points[:min_size, :]
        padded_batch.append((points, target))
    
    # 使用默认的 collate_fn 处理填充后的批次
    return torch.utils.data.dataloader.default_collate(padded_batch)

# provider
def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.333):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc

# 保存模型参数和缓冲区为 .txt 文件
def save_model_params_and_buffers_to_txt(model, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # 保存所有参数
    for name, param in model.named_parameters():
        np.savetxt(os.path.join(directory, f'{name}.txt'), param.detach().cpu().numpy().flatten())
    
    # 保存所有缓冲区
    for name, buffer in model.named_buffers():
        np.savetxt(os.path.join(directory, f'{name}.txt'), buffer.detach().cpu().numpy().flatten())

def main():
    # 创建数据集实例
    data_path = './data'
    time_start = time.time()

    train_dataset = PointCloudDataset(root=data_path, split='train', enhance=False)
    train_test_dataset = PointCloudDataset(root=data_path, split='train')

    # 创建 DataLoader 实例
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, drop_last=True)
    #train_split_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, drop_last=True)
    train_test_dataloader = DataLoader(train_test_dataset, batch_size=1, shuffle=False)

    print("finish DATA LOADING")

    # MODEL LOADING
    classifier = get_model(num_class)
    criterion = get_loss()
    classifier.apply(inplace_relu)

    classifier = classifier.cuda()
    criterion = criterion.cuda()

    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.003, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    # 初始化 GradScaler
    scaler = GradScaler()

    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    print("finish MODEL LOADING")

    # TRANING
    print("start TRAINING")
    for epoch in range(total_epoch):
        print('Epoch %d (%d/%s):' % (epoch, epoch, total_epoch))

        mean_correct = []
        classifier.train()

        for batch_id, (points, target) in enumerate(train_dataloader, 0):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = random_point_dropout(points)
            #points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            points, target = points.cuda(), target.cuda()

            # 使用 autocast 进行前向传播
            with autocast():
                pred, trans_feat = classifier(points)
                loss = criterion(pred, target.long(), trans_feat)

            # 使用 GradScaler 进行反向传播和梯度更新
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))

        scheduler.step()

        train_instance_acc = np.mean(mean_correct)
        print('Train Instance Accuracy: %f' % train_instance_acc)

        #path_model = os.path.join(script_dir, f'model{epoch}.pth')
        #torch.save(classifier.state_dict(), path_model)

        #if epoch % 20 == 0:
        #    save_model_params_and_buffers_to_txt(classifier, script_dir)

        with torch.no_grad():
            #instance_acc = test(classifier.eval(), train_split_dataloader, num_class=num_class)
            #print('Test in Train Instance Accuracy: %f' % instance_acc)

            if epoch > 10:
                test_acc = test(classifier.eval(), train_test_dataloader, num_class=num_class)
                print('Trainset Test Accuracy: ', test_acc)
                if test_acc > best_class_acc:
                    save_model_params_and_buffers_to_txt(classifier, script_dir)
                    best_class_acc = test_acc

        time_used = time.time() - time_start
        print('time used: ', time_used)
        if time_used > 78 * 60:
            print("time use all")
            break
    print("finish TRAINING")

if __name__ == '__main__':
    main()