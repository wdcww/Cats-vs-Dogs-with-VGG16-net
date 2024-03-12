import torch
from numpy import random
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import mydataset  # 从dataset.py导入
from net import NET  # 从net.py导入

# 1.打开txt.py得到的“data.txt”,用 f.readlines()得到一个list
with open("data.txt", "r") as f:
    lines = f.readlines()

# 2.打乱lines的顺序,为数据集做准备
random.shuffle(lines)

# 3.打算用所有数据的1/5 验证集
val_number = int(len(lines) * 0.2)
train_number = len(lines) - val_number

# 4.定义transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 假设使用224x224大小的图片输入
    transforms.ToTensor(),  # 将图片转换为张量
])

# 5.dataset
val_dataset = mydataset(lines[:val_number], transform=transform)
train_dataset = mydataset(lines[val_number:], transform=transform)
print(f"train集长度{len(train_dataset)},val集的长度{len(val_dataset)}")

# 6.DataLoader 注意 "D"和"L"都是大写
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 7.device
# device = torch.device("cuda:0")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 8.net ####################################################################
net = NET()

# ## 如果想要使用vgg16的预训练权重,【解开下面紧接着的这行的注释】,并且【在net.py里面使用“没有normalization的net”】
# net.load_state_dict(torch.load(r"vgg16-397923af.pth"))


# ## 我在net.py定义的网络，最后一个nn.Linear的out_features = 1000(参考VGG16结构)
# ## 这里是2分类,所以要让最后一个nn.Linear的out_features = 2,下面这行添加从1000->2
net.classifier.add_module("my", nn.Linear(1000, 2))
# print(net)
net.to(device)
# ###########################################################################


# 9.优化器、lr
# 这一部分写在了train.py


# 10.损失函数
my_loss = nn.CrossEntropyLoss().to(device)



# PS:
#    整个流程只把 net、dataloader里的数据(指image,label)、损失函数my_loss 这三者传到了GPU
#       net和my_loss在这个.py文件里直接.to(device),
#    dataloader里的数据在train.py里的训练里放到gpu
