from torch.utils.data import Subset,DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision import datasets, transforms

def get_data_loader(node_id, num_nodes=4, batch_size=32):
    """根据节点ID返回划分后的数据加载器"""
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # MNIST示例
    ])
    
    # 加载完整数据集
    full_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # 划分策略：按标签范围分配（示例）
    labels_per_node = 10 // num_nodes
    start_label = node_id * labels_per_node
    end_label = (node_id + 1) * labels_per_node
    
    selected_indices = [
        i for i, (_, label) in enumerate(full_dataset)
        if start_label <= label < end_label
    ]
    
    # 创建子集和数据加载器
    subset = Subset(full_dataset, selected_indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True)
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
