import torch
# torch.cuda.get_device_name()

if __name__=='__main__':
    # 获取GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"GPU数量：{num_gpus}")

    # 获取每个GPU的名称和状态
    for i in range(num_gpus):
        gpu = torch.device(f"cuda:{i}")
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}, 可用状态：{torch.cuda.is_available()}")