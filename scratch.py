import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from sklearn.model_selection import LeaveOneOut
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader, Subset
import scipy.io as io
import os
import numpy as np
import matplotlib.pyplot as plt
import time

class RamanDataset(Dataset):
    def __init__(self, data, wavelengths, labels):
        self.data = data
        self.wavelengths = wavelengths
        # 确保labels是一维张量
        self.labels = labels

    def __getitem__(self, index):
        spectrum = torch.tensor(self.data[index], dtype=torch.float32)
        label = self.labels[index]  # 直接使用标签
        return spectrum, label

    def __len__(self):
        return len(self.data)


def preprocess_excel(excel_path, header=0):
    # 读取Excel文件
    df = pd.read_excel(excel_path)

    # 提取样本名和光谱数据
    sample_names = df['Sample']  # 假设样本名在列名为'sample'的列
    wavelengths = np.arange(250, 3001, 1)  # 您的波长范围
    spectra_data = df.drop(columns=['Sample']).values.T  # 除去样本名后的光谱数据

    # 创建RamanDataset实例
    labels = torch.arange(len(sample_names), dtype=torch.long)  # 标签从0开始
    dataset = RamanDataset(spectra_data, wavelengths, labels)
    # 创建RamanDataset实例，包含数据、波长和标签
    return dataset, sample_names

def custom_loocv(dataset, num_classes):
    """
    自定义LOOCV逻辑，适用于每个样本可能有多个光谱的情况。
    sample_indices: 样本索引数组。
    spectra: 所有样本的峰值数据数组。
    """
    sample_indices = list(range(len(dataset)))
    evaluation_results = []
    for i in sample_indices:

        train_subsampler = Subset(dataset, sample_indices[:i] + sample_indices[i + 1:])
        test_subsampler = Subset(dataset, [sample_indices[i]])

        train_loader = DataLoader(train_subsampler, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_subsampler, batch_size=1, shuffle=False)



    # 这里需要您定义模型、损失函数、优化器，并实现训练和评估逻辑
        # model = YourModel()
        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=0.001)
        # train_model(model, train_loader, optimizer, criterion)
        # test_model(model, test_loader, criterion)
        # result = evaluate_model(model, test_loader)
        # eval_results.append(result)

    return evaluation_results

# 定义网络
class MyNet(nn.Module):
    def __init__(self, num_classes):
        super(MyNet, self).__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=4, stride=2, padding=1)  # shape (1*1*224)-->(16*1*112)输入通道数为1，输出通道数为16，卷积核大小为4×1，步长为2。
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1)  # (16*1*112)-->(32*1*56)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1)  # (32*1*56)-->(64*1*28)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)  # (64*1*28)-->(128*1*14)
        self.bn4 = nn.BatchNorm1d(128)

        # self.conv5 =nn.Conv1d(128,256,kernel_size=4,stride=2,padding=1)#(128*1*14)-->(256*1*7)
        # self.bn5=nn.BatchNorm1d(256)

        # self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num_classes)

    def forward(self, x):

        x = x.unsqueeze(1)  # shape (batch_size, 1, sequence_length)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        # x=F.relu(self.conv5(x))
        # x=self.bn5(x)

        x = x.view(x.size(0), -1)
        x = x.view(-1, 128 * 14)
        x = F.relu(self.fc1(x))
        # x=self.drop(x)
        x = F.relu(self.fc2(x))
        # x=self.drop(x)
        x = F.relu(self.fc3(x))
        # x=self.drop(x)
        x = F.relu(self.fc4(x))
        self.feature = x
        # x=self.drop(x)
        x = self.fc5(x)

        # x = x.squeeze(-1)
        return x

    def weight_init(self, module=None):
        if module is None:
            module = self  # 如果没有传入模块，则默认使用当前模块

        for m in module.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)  # 选择一种初始化方法
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data)


# LOOCV逻辑
# Excel文件路径
excel_path = './modified_combined_data.xlsx'  # 替换为您的Excel文件路径
dataset, sample_names = preprocess_excel(excel_path)

# 初始化 data_by_sample 字典
data_by_sample = {}

# 假设样本名是唯一的，我们将每个样本名映射到其索引
for index, sample_name in enumerate(sample_names):
    if sample_name not in data_by_sample:
        data_by_sample[sample_name] = []
    data_by_sample[sample_name].append(index)

# 执行自定义LOOCV
evaluation_results = custom_loocv(dataset, num_classes=899)
data_by_sample = {name: [] for name in sample_names}  # 初始化数据索引字典

for index, sample_name in enumerate(sample_names):
    data_by_sample[sample_name].append(index)

for sample_name, indices in data_by_sample.items():
    if len(indices) > 1:
        # 如果样本有多个光谱数据，实现留一交叉验证
        for i in range(len(indices)):
            train_indices = indices[:i] + indices[i+1:]
            test_indices = [indices[i]]
    else:
        # 如果样本只有一条光谱数据，则该光谱既用作训练也用作测试
        train_indices = [indices[0]]
        test_indices = [indices[0]]

    # 创建训练和测试的Subset
    train_subsampler = Subset(dataset, train_indices)
    test_subsampler = Subset(dataset, test_indices)

    # 创建训练和测试的DataLoader
    train_loader = DataLoader(train_subsampler, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_subsampler, batch_size=1, shuffle=False)

    # 定义模型、损失函数、优化器，并实现训练和评估逻辑
    model = MyNet(num_classes=899)
    model.apply(model.weight_init)  # 应用权重初始化
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    model.train()
    for epoch in range(52):  # 进行52个训练周期
        for batch in train_loader:
            # 解包DataLoader返回的元组，获取inputs和labels
            inputs, labels = batch  # 解包操作

            # 确保输入数据是浮点数类型
            inputs = inputs.float()

            # 确保标签是二维张量，批量大小与模型输出匹配
            if labels.dim() == 1:
                labels = labels.view(-1, 1)  # 将一维标签张量转换为二维

            optimizer.zero_grad()
            outputs = model(inputs)
            # 确保传递给损失函数的outputs和labels具有正确的形状
            loss = criterion(outputs, labels.squeeze(1))  # 移除labels的单维度
            loss.backward()
            optimizer.step()

    # 使用适当的标签
    true_labels = [sample_name for _ in test_indices]  # 假设样本名用作真实标签
    # 评估模型
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in test_loader:
            inputs, labels = batch
            inputs = inputs.float()  # 确保输入数据是浮点数类型
            if labels.dim() == 0:
                labels = labels.view(1, 1)  # 确保标签是二维张量
            else:
                labels = labels.view(-1, 1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze(1)).sum().item()
        accuracy = 100 * correct / total
        evaluation_results.append(accuracy)

# 计算评估结果的平均值
average_accuracy = sum(evaluation_results) / len(evaluation_results)
print(f"Custom LOOCV Average Accuracy: {average_accuracy:.2f}%")

# 我的数据集是每一行一个样本，第一行第一列内容是sample，第二列到最后一列为从250到3000纳米，间隔为1，第二行到最后一行每一行为一个样本，这些行的第一列为样本名字，第二列到最后一列为该样本在不同波长处的峰值 如何修改代码，数据集是excel表，不同的样本的拉曼光谱数据各有不同，有的样本有一条，有的样本有两三条，采用留一交叉法，数据有两条及以上的，留一条作为测试集，剩下的作为训练集，数据集只有一条的，又作为训练集又作为测试集，从而进行定性判别，一共有1121条数据，在这些数据中有899个类别，如何修改代码？

















# # model.weight_init()
# # model.apply(weights_init)
# print(model)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
# # optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
# start = time.time()
# # 开始训练
# train_losses = []
# epoch_list = []
# train_accuracy_list = []
# test_losses = []
# test_accuracy_list = []
# for epoch in range(100):  # epoch在100至300之间
#
#     train_loss = 0.
#     test_loss = 0.
#     train_num_correct = 0
#     test_num_correct = 0
#     for i, data in enumerate(train_loader):
#         inputs, labels = data
#         optimizer.zero_grad()
#
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item()
#         pred = outputs.argmax(dim=1)
#         train_num_correct += torch.eq(pred, labels).sum().float().item()
#         # min_train_num_correct=0
#         # if train_num_correct>=min_train_num_correct:
#         # min_rain_num_correct=train_num_correct
#         out_feature = model.feature
#
#     with torch.no_grad():
#         outputs_test_list = []
#         for i, data in enumerate(test_loader):
#             inputs_test, label_test = data
#             outputs_test = model(inputs_test)
#             # outputs_test_list.append(outputs_test.numpy())
#             loss_test = criterion(outputs_test, label_test)
#
#             test_loss += loss_test.item()
#             pred_test = outputs_test.argmax(dim=1)
#             test_num_correct += torch.eq(pred_test, label_test).sum().float().item()
#
#     print("Epoch: {}\t Train Loss: {:.6f}\t Train Acc: {:.6f}\t Test Loss: {:.6f}\t Test Acc: {:.6f}".format(epoch,
#                                                                                                              train_loss / len(
#                                                                                                                  train_loader),
#                                                                                                              train_num_correct / len(
#                                                                                                                  train_loader.dataset),
#                                                                                                              test_loss / len(
#                                                                                                                  test_loader),
#                                                                                                              test_num_correct / len(
#                                                                                                                  test_loader.dataset)))
#     # print("Epoch: {}\t Train Loss: {:.6f}\t \t Test Loss: {:.6f}".format(epoch,train_loss/len(train_loader),test_loss/len(test_loader)))
#
#     train_losses.append(train_loss / len(train_loader))
#     train_accuracy_list.append(train_num_correct / len(train_loader.dataset))
#     test_losses.append(test_loss / len(test_loader))
#     test_accuracy_list.append(test_num_correct / len(test_loader.dataset))
#     epoch_list.append(epoch)
#
# print("finished training")
# print('time = %2dm:%2ds' % ((time.time() - start) // 60, (time.time() - start) % 60))
# # 绘制loss和accuracy曲线
# plt.figure(figsize=(10, 8))
# plt.plot(epoch_list, train_losses)
# plt.plot(epoch_list, test_losses)
# plt.xlabel('Epoch', fontsize="x-large")
# plt.ylabel('Loss and Accuracy', fontsize='x-large')
# plt.plot(epoch_list, train_accuracy_list)
# plt.plot(epoch_list, test_accuracy_list)
# plt.legend(['Train Loss', 'Test Loss', 'Train Accuracy', 'Test Accuracy'], fontsize="x-large", loc='upper right')
# # plt.savefig('loss and accuracy_cv4.png',bbox_inches='tight',dpi=300,pad_inches=0.1)
# plt.show()
#
#
#
# # 计算评估结果的平均值
# average_result = sum(evaluation_results) / len(evaluation_results)
# print(f"Custom LOOCV Average Result: {average_result}")
#
