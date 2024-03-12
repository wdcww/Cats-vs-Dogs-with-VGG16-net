import time
import numpy as np
import torch
from matplotlib import pyplot as plt

from prepare_train import device, net, my_loss, val_dataset, train_dataset, train_dataloader, val_dataloader

# #opt优化器、lr########
learn_rate = 1e-4
opt = torch.optim.Adam(net.parameters(), lr=learn_rate)
# opt = torch.optim.Adam(net.parameters(), lr=learn_rate,weight_decay=0.01)  # 权重衰退
# #####################






# TRAIN & VAL
# epochs = int(input('输入轮数：'))
epochs = 2

print("一共有" + str(epochs) + "轮")

# 下面的空列表用来放画图的值
list_train_losses = []
list_val_losses = []
acc_tra = []
acc_val = []

for epoch in range(epochs):

    # # ###变学习率
    # # 0、1、2、3、4、5、6、7、8、9 ，learn_rate=1e-4
    #
    # if epoch>9:
    #     # 10、11、12、13、14，learn_rate=1e-5
    #    learn_rate=1e-5
    #
    # if epoch>14:
    #     # 15、16、17、18、19、20，learn_rate=1e-6
    #    learn_rate = 1e-6

    start_time = time.time()

    train_losses = 0.0
    val_losses = 0.0
    val_accuracy = 0.0
    train_accuracy = 0.0

    # train ###########################
    for data in train_dataloader:
        image, label = data
        image = image.to(device)
        label = label.to(device)

        # 优化器梯度设置为0
        opt.zero_grad()

        # 数据送入网络
        output = net(image)

        acc = (output.argmax(1) == label).sum()
        train_accuracy += acc

        # 预测值与label算loss
        train_loss = my_loss(output, label)

        # 计算loss关于权重的梯度
        train_loss.backward()

        # 更新权重参数
        opt.step()

        # 累加每个小批次的loss到总的train_losses
        train_losses += train_loss

    end_time_1 = time.time()

    # 这一轮总的train_losses放入 list_train_losses:list
    list_train_losses.append(train_losses)

    # 计算这一轮训练的accuracy,并放入 acc_tra:list
    train_accuracy = train_accuracy / len(train_dataset)
    acc_tra.append(train_accuracy)

    # val #########################
    for data in val_dataloader:
        image, label = data

        with torch.no_grad():  # 验证不算梯度
            image = image.to(device)
            label = label.to(device)

            out = net(image)  # 投入网络

            accuracy = (out.argmax(1) == label).sum()  # 计算一下这一小批次预测对的数量
            val_accuracy += accuracy  # 总计所有批次预测对的数量

            # # 验证集好像不看loss吧。
            # test_loss = my_loss(out, label)
            # val_losses += test_loss

    end_time_2 = time.time()

    # list_val_losses.append(val_losses) # 用于画图的

    val_accuracy = val_accuracy / len(val_dataset)
    acc_val.append(val_accuracy)


    print("第{}轮,学习率{},训练用时{:.2}s,测试用时{:.2}s".format(epoch + 1, learn_rate, end_time_1 - start_time,
                                                                 end_time_2 - end_time_1))
    print("      训练集总loss：{}".format(train_losses))
    # print("第{}轮,验证集总loss：{}".format(epoch + 1,val_losses))
    print("      训练集上的精度：{:.1%}".format(train_accuracy))
    print("      验证集上的精度：{:.1%}".format(val_accuracy))


# ## 最后保存一下模型
# torch.save(net,"./model/{}.pth".format(epoch+1))
# print("模型已保存")


# 下面画图的
acc_tra = [item.item() for item in acc_tra]
acc_val = [item.item() for item in acc_val]
list_train_losses = [item.item() for item in list_train_losses]
# list_val_losses = [item.item() for item in list_val_losses]
#


fig = plt.figure()

# 添加第一个子图，设置标题和坐标轴标签
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(list(range(1, len(acc_tra) + 1)), np.array(acc_tra), color='red', marker='o', label='acc_tra')
ax1.plot(list(range(1, len(acc_val) + 1)), np.array(acc_val), color='black', marker='o', label='acc_val')
plt.legend()

# 添加第二个子图，设置标题和坐标轴标签
ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(list(range(1, len(list_train_losses) + 1)), list_train_losses, color='blue', marker='o', label='train_loss')
plt.legend()

# 保存图片
plt.savefig(r"p.png")
plt.close()


#  2023.12.1
# def train_val(epochs:int, device:torch.device, opt, loss, net,
#               train_dataset_len:int,
#               val_dataset_len:int,
#               train_dataloader, val_dataloader,
#               net_save=False):
#     """
#     这是一个封装了2层for循环框架过程的函数
#
#     形参依次是：
#       epochs：轮数
#
#       device：设备
#
#       opt：优化器
#
#       loss：损失函数
#
#       net：模型
#
#       train_dataset_len：训练数据集长度
#
#       val_dataset_len：验证数据集长度
#
#       train_dataloader：训练集DataLoader
#
#       val_dataloader：验证集DataLoader
#
#       net_save：默认为False
#
#     return
#         训练集loss列表、验证集loss列表、训练集精度列表、验证集精度列表。
#     """
#
#     print(f"你的训练数据集长度{train_dataset_len}验证数据集长度{val_dataset_len}")
#     print(f"device：{device}")
#
#     # 下面两个空列表用来放画图的值的
#     list_train_losses = []
#     list_val_losses = []
#     acc_tra = []
#     acc_val = []
#
#     for epoch in range(epochs):
#         start_time = time.time()
#
#         train_losses = 0.0
#         val_losses = 0.0
#         val_accuracy = 0.0
#         train_accuracy = 0.0
#
#         for data in train_dataloader:
#             image, label = data
#             image = image.to(device)
#             label = label.to(device)
#
#             # 优化器梯度设置为0
#             opt.zero_grad()
#
#             # 数据送入网络
#             output = net(image)
#
#             acc = (output.argmax(1) == label).sum()
#             train_accuracy += acc
#
#             # 预测值与label算loss
#             train_loss = loss(output, label)
#
#             # 计算loss关于权重的梯度
#             train_loss.backward()
#
#             # 更新权重参数
#             opt.step()
#
#             # 累加每个小批次的loss到总的train_losses
#             train_losses += train_loss
#
#         train_accuracy = train_accuracy / train_dataset_len
#         acc_tra.append(train_accuracy)
#         end_time_1 = time.time()
#
#         list_train_losses.append(train_losses)
#
#         # 测试
#         for data in val_dataloader:
#             image, label = data
#
#             with torch.no_grad():  # 验证不算梯度
#                 image = image.to(device)
#                 label = label.to(device)
#
#                 out = net(image)  # 投入网络
#
#                 accuracy = (out.argmax(1) == label).sum()  # 计算一下这一小批次预测对的数量
#                 val_accuracy += accuracy  # 总计所有批次预测对的数量
#
#
#                 test_loss = loss(out, label)
#                 val_losses += test_loss
#
#         end_time_2 = time.time()
#
#         val_accuracy = val_accuracy / val_dataset_len
#         acc_val.append(val_accuracy)
#         list_val_losses.append(val_losses)  # 用于画图的
#
#         print("第{}轮,训练用时{:.2}s,测试用时{:.2}s".format(epoch + 1, end_time_1 - start_time,end_time_2 - end_time_1))
#         print("      训练集总loss：{}".format(train_losses))
#         print("      验证集总loss：{}".format(epoch + 1,val_losses))
#         print("      训练集上的精度：{:.1%}".format(train_accuracy))
#         print("      验证集上的精度：{:.1%}".format(val_accuracy))
#     if net_save:
#         ## 最后保存一下模型
#         torch.save(net,"./model/{}.pth".format(epoch+1))
#         print("模型已保存")
#
#
#     acc_tra = [item.item() for item in acc_tra]
#     acc_val = [item.item() for item in acc_val]
#     list_train_losses = [item.item() for item in list_train_losses]
#     list_val_losses = [item.item() for item in list_val_losses]
#
#
#     return acc_tra,acc_val,list_train_losses,list_val_losses
