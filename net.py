import torch.nn as nn
import torch
from torch import flatten


# BN #############################################################################
class NET(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.features = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=64, padding=1, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # 参数inplace = True是指原地进行操作，操作完成后覆盖原来的变量

            nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),  # Cin*224*224 变成了 Cout*112*112

            nn.Conv2d(in_channels=64, out_channels=128, padding=1, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),  # 参数inplace = True是指原地进行操作，操作完成后覆盖原来的变量

            nn.Conv2d(in_channels=128, out_channels=128, padding=1, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),  # Cin*112*112变成Cout*56*56

            nn.Conv2d(in_channels=128, out_channels=256, padding=1, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),  # 参数inplace = True是指原地进行操作，操作完成后覆盖原来的变量

            nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),  # Cin*56*56变成 Cout*28*28

            nn.Conv2d(in_channels=256, out_channels=512, padding=1, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),  # Cin*28*28 变成 Cout*14*14

            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),  # Cin*14*14 变成 Cout*7*7
        )

        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),  # 512 * 7 * 7 -> 4096*1*1
            nn.BatchNorm1d(4096),
            nn.ReLU(True),

            nn.Dropout(),  # 防止过拟合


            nn.Linear(4096, 4096),  # 4096*1*1 -> 4096*1*1
            nn.BatchNorm1d(4096),
            nn.ReLU(True),

            nn.Dropout(),  # 防止过拟合


            nn.Linear(4096, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(True),

            nn.Dropout(),  # 防止过拟合
        )

    def forward(self, x):
        x = self.features(x)
        x = flatten(x, 1)
        x = self.classifier(x)
        return x




# ## LN #############################################################################
# class NET(nn.Module):
#     def __init__(self):
#         nn.Module.__init__(self)
#         self.features = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64, padding=1, kernel_size=3),
#             nn.LayerNorm([64, 224, 224], eps=1e-05, elementwise_affine=True),
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3),
#             nn.LayerNorm([64, 224, 224], eps=1e-05, elementwise_affine=True),
#             nn.ReLU(),
#
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Cin*224*224 变成了 Cout*112*112
#
#             nn.Conv2d(in_channels=64, out_channels=128, padding=1, kernel_size=3),
#             nn.LayerNorm([128, 112, 112], eps=1e-05, elementwise_affine=True),
#             nn.ReLU(),  # 参数inplace = True是指原地进行操作，操作完成后覆盖原来的变量
#
#             nn.Conv2d(in_channels=128, out_channels=128, padding=1, kernel_size=3),
#             nn.LayerNorm([128, 112, 112], eps=1e-05, elementwise_affine=True),
#             nn.ReLU(),
#
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Cin*112*112变成Cout*56*56
#
#             nn.Conv2d(in_channels=128, out_channels=256, padding=1, kernel_size=3),
#             nn.LayerNorm([256, 56, 56], eps=1e-05, elementwise_affine=True),
#             nn.ReLU(),  # 参数inplace = True是指原地进行操作，操作完成后覆盖原来的变量
#
#             nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3),
#             nn.LayerNorm([256, 56, 56], eps=1e-05, elementwise_affine=True),
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3),
#             nn.LayerNorm([256, 56, 56], eps=1e-05, elementwise_affine=True),
#             nn.ReLU(),
#
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Cin*56*56变成 Cout*28*28
#
#             nn.Conv2d(in_channels=256, out_channels=512, padding=1, kernel_size=3),
#             nn.LayerNorm([512, 28, 28], eps=1e-05, elementwise_affine=True),
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3),
#             nn.LayerNorm([512, 28, 28], eps=1e-05, elementwise_affine=True),
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3),
#             nn.LayerNorm([512, 28, 28], eps=1e-05, elementwise_affine=True),
#             nn.ReLU(),
#
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Cin*28*28 变成 Cout*14*14
#
#             nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3),
#             nn.LayerNorm([512, 14, 14], eps=1e-05, elementwise_affine=True),
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3),
#             nn.LayerNorm([512, 14, 14], eps=1e-05, elementwise_affine=True),
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3),
#             nn.LayerNorm([512, 14, 14], eps=1e-05, elementwise_affine=True),
#             nn.ReLU(),
#
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Cin*14*14 变成 Cout*7*7
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Linear(25088, 4096),  # 512 * 7 * 7 -> 4096*1*1
#             nn.LayerNorm(4096, eps=1e-05, elementwise_affine=True),
#             nn.ReLU(True),
#
#             nn.Dropout(),  # 防止过拟合
#
#             nn.Linear(4096, 4096),  # 4096*1*1 -> 4096*1*1
#             nn.LayerNorm(4096, eps=1e-05, elementwise_affine=True),
#             nn.ReLU(True),
#
#             nn.Dropout(),  # 防止过拟合
#
#             nn.Linear(4096, 1000),
#             nn.LayerNorm(1000, eps=1e-05, elementwise_affine=True),
#             nn.ReLU(True),
#
#             nn.Dropout(),  # 防止过拟合
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = flatten(x, 1)
#         x = self.classifier(x)
#         return x



# #   GN在通道上进行分组，每个组将包含 num_channels / num_groups 个通道。
# # #### GN ####################################################################
# class NET(nn.Module):
#     def __init__(self):
#         nn.Module.__init__(self)
#         self.features = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64, padding=1, kernel_size=3),
#             nn.GroupNorm(num_groups=4, num_channels=64,eps=1e-5,affine=True), # 每个组将包含 num_channels / num_groups 个通道。
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3),
#             nn.GroupNorm(num_groups=4, num_channels=64,eps=1e-5,affine=True),
#             nn.ReLU(),
#
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Cin*224*224 变成了 Cout*112*112
#
#             nn.Conv2d(in_channels=64, out_channels=128, padding=1, kernel_size=3),
#             nn.GroupNorm(num_groups=4, num_channels=128,eps=1e-5,affine=True),
#             nn.ReLU(),  # 参数inplace = True是指原地进行操作，操作完成后覆盖原来的变量
#
#             nn.Conv2d(in_channels=128, out_channels=128, padding=1, kernel_size=3),
#             nn.GroupNorm(num_groups=4, num_channels=128,eps=1e-5,affine=True),
#             nn.ReLU(),
#
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Cin*112*112变成Cout*56*56
#
#             nn.Conv2d(in_channels=128, out_channels=256, padding=1, kernel_size=3),
#             nn.GroupNorm(num_groups=4, num_channels=256,eps=1e-5,affine=True),
#             nn.ReLU(),  # 参数inplace = True是指原地进行操作，操作完成后覆盖原来的变量
#
#             nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3),
#             nn.GroupNorm(num_groups=4, num_channels=256,eps=1e-5,affine=True),
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3),
#             nn.GroupNorm(num_groups=4, num_channels=256, eps=1e-5, affine=True),
#             nn.ReLU(),
#
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Cin*56*56变成 Cout*28*28
#
#             nn.Conv2d(in_channels=256, out_channels=512, padding=1, kernel_size=3),
#             nn.GroupNorm(num_groups=4, num_channels=512,eps=1e-5,affine=True),
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3),
#             nn.GroupNorm(num_groups=4, num_channels=512,eps=1e-5,affine=True),
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3),
#             nn.GroupNorm(num_groups=4, num_channels=512,eps=1e-5,affine=True),
#             nn.ReLU(),
#
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Cin*28*28 变成 Cout*14*14
#
#             nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3),
#             nn.GroupNorm(num_groups=4, num_channels=512,eps=1e-5,affine=True),
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3),
#             nn.GroupNorm(num_groups=4, num_channels=512,eps=1e-5,affine=True),
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3),
#             nn.GroupNorm(num_groups=4, num_channels=512,eps=1e-5,affine=True),
#             nn.ReLU(),
#
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Cin*14*14 变成 Cout*7*7
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Linear(25088, 4096),  # 512 * 7 * 7 -> 4096*1*1
#             nn.GroupNorm(num_groups=4, num_channels=4096,eps=1e-5,affine=True),
#             nn.ReLU(True),
#
#             nn.Dropout(),  # 防止过拟合
#
#             nn.Linear(4096, 4096),  # 4096*1*1 -> 4096*1*1
#             nn.GroupNorm(num_groups=4, num_channels=4096,eps=1e-5,affine=True),
#             nn.ReLU(True),
#
#             nn.Dropout(),  # 防止过拟合
#
#             nn.Linear(4096, 1000),
#             nn.GroupNorm(num_groups=4, num_channels=1000,eps=1e-5,affine=True),
#             nn.ReLU(True),
#
#             nn.Dropout(),  # 防止过拟合
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = flatten(x, 1)
#         x = self.classifier(x)
#         return x



# # 早期没有normalization的net
# class NET(nn.Module):
#     def __init__(self):
#         nn.Module.__init__(self)
#         self.features = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64, padding=1, kernel_size=3),
#
#             nn.ReLU(),  # 参数inplace = True是指原地进行操作，操作完成后覆盖原来的变量
#
#             nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3),
#
#             nn.ReLU(),
#
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Cin*224*224 变成了 Cout*112*112
#
#             nn.Conv2d(in_channels=64, out_channels=128, padding=1, kernel_size=3),
#
#             nn.ReLU(),  # 参数inplace = True是指原地进行操作，操作完成后覆盖原来的变量
#
#             nn.Conv2d(in_channels=128, out_channels=128, padding=1, kernel_size=3),
#
#             nn.ReLU(),
#
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Cin*112*112变成Cout*56*56
#
#             nn.Conv2d(in_channels=128, out_channels=256, padding=1, kernel_size=3),
#
#             nn.ReLU(),  # 参数inplace = True是指原地进行操作，操作完成后覆盖原来的变量
#
#             nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3),
#
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=256, out_channels=256, padding=1, kernel_size=3),
#
#             nn.ReLU(),
#
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Cin*56*56变成 Cout*28*28
#
#             nn.Conv2d(in_channels=256, out_channels=512, padding=1, kernel_size=3),
#
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3),
#
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3),
#
#             nn.ReLU(),
#
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Cin*28*28 变成 Cout*14*14
#
#             nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3),
#
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3),
#
#             nn.ReLU(),
#
#             nn.Conv2d(in_channels=512, out_channels=512, padding=1, kernel_size=3),
#
#             nn.ReLU(),
#
#             nn.MaxPool2d(kernel_size=2, stride=2),  # Cin*14*14 变成 Cout*7*7
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Linear(25088, 4096),  # 512 * 7 * 7 -> 4096*1*1
#             nn.ReLU(True),
#
#             nn.Dropout(),  # 防止过拟合
#
#             nn.Linear(4096, 4096),  # 4096*1*1 -> 4096*1*1
#             nn.ReLU(True),
#
#             nn.Dropout(),  # 防止过拟合
#
#             nn.Linear(4096, 1000),
#             nn.ReLU(True),
#
#             nn.Dropout(),  # 防止过拟合
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = flatten(x, 1)
#         x = self.classifier(x)
#         return x




if __name__ == '__main__':
    net = NET()
    print(net)
    in_data = torch.ones(5, 3, 224, 224)
    # print(in_data)
    out = net(in_data)

    print(out)
