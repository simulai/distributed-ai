from torch.utils.data import Subset
from torchvision.datasets import CIFAR10

def split_dataset(node_id, num_nodes=4):
    """按空间位置划分CIFAR-10数据集"""
    full_dataset = CIFAR10(root='./data', train=True, download=True)
    
    # 将32x32图像分为4个16x16区域
    split_rules = [
        (slice(0,16), slice(0,16)),   # 左上
        (slice(0,16), slice(16,32)),  # 右上
        (slice(16,32), slice(0,16)),  # 左下 
        (slice(16,32), slice(16,32))  # 右下
    ]
    
    selected_indices = [
        i for i, (img, _) in enumerate(full_dataset)
        if img.crop(split_rules[node_id]).mean() > threshold  # 示例划分条件
    ]
    
    return Subset(full_dataset, selected_indices)
