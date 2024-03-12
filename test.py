from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
import torch.nn.functional

from txt import animal_class

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 假设使用224x224大小的图片输入
    transforms.ToTensor(),  # 将图片转换为张量
])



image_path=r"test1.png"

i=Image.open(image_path)
if i.mode != 'RGB':
    i = i.convert('RGB')
    i.save(image_path)

img=Image.open(image_path)
image=transform(img)
image = torch.reshape(image, (1, 3, 224, 224))

model=torch.load(r"")    # 这里写训练完保存的网络权重地址,
#                        # train.py保存的格式是torch.save(net,"./model/{}.pth".format(epoch+1))
image=image.to('cuda')



model.eval()
# model.eval() 将模型切换到测试模式，
# 确保在测试时使用移动平均值进行批标准化。请确保在测试时将模型切换到 eval 模式，以确保 BN 层的正确行为。
# 在训练时，BN使用批次内的统计信息来进行标准化。
# 但是在测试时，通常会使用在训练期间累积的移动平均值来进行标准化。



out = model(image)

print("从模型出来的out",out)
out=torch.nn.functional.softmax(out,dim=1)
print("经过softmax的out",out)

a=int(out.argmax(1))

out=out.data.cpu().numpy()
# animal_class=['cat','dog','bee','ant']
plt.suptitle("Classes:{},P={:.1%}".format(animal_class[a],out[0,a]))
plt.imshow(img)
plt.show()



