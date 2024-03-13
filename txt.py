import os

animal_class = ['cat', 'dog']

root_dir = r"raw_data"  # 存放 train、val所有的图片 的路径

if __name__ == '__main__':
    f = open("data.txt", mode="w")  # 以write的方式打开data.txt
    items = os.listdir(root_dir)  # 返回一个包含此目录中所有子目录名称的列表list
    # 上面这行是很关键的os.listdir()函数，能够获得你输入的path下面所有文件的名字的列表
    for item in items:  # for循环去访问list中的每个元素

        if item not in animal_class:
            continue

        # label
        label = animal_class.index(item)  # animal_class中的类别转换为类别索引号

        # 图片名字
        pic_path = os.path.join(root_dir, item)
        pic_name_list = os.listdir(pic_path)  # 返回一个包含此目录下所有图片名称的列表

        # 写入txt
        for every_pic_name in pic_name_list:
            f.write(str(label) + ";" + "%s%s%s" % (pic_path, '/', every_pic_name))
            f.write("\n")
    f.close()