from PIL import Image
from torch.utils.data import Dataset


class mydataset(Dataset):
    """
    这是一个继承了Dataset类的子类，
    需要去传入你的包含有图片名称的list以及transform，
    list可以是通过切片的一段sub_list，这样可以有不同的sub_list对应train_dataset和 test_dataset
    """
    def __init__(self,sub_list,transform=None):
        self.sub_list = sub_list
        self.transform = transform

    def __len__(self):
        return len(self.sub_list)

    def __getitem__(self, index):
        # 标签
        label = int(self.sub_list[index].split(";")[0])

        # 图片
        image_path = self.sub_list[index].split(";")[1].strip('\n')
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            image.save(image_path)

        # transform
        image = self.transform(image)

        return image,label