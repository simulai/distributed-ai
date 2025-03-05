import torch
import yaml
from multiprocessing import Process, Queue
from .protocol import compress_grad, decompress_grad

class VirtualNode(Process):
    def __init__(self, node_id, comm_queue, config_path):
        super().__init__()
        self.node_id = node_id
        self.comm_queue = comm_queue
        with open(config_path) as f:
            self.config = yaml.safe_load(f)['communication']
        
        # 初始化模型和数据
        self.model = MiniCNN()  # 自定义小模型
        self.train_loader = get_data_loader(node_id)  # 数据划分方法

    def run(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        for epoch in range(20):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                # 本地训练
                output = self.model(data)
                loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                
                # 触发通信条件
                if batch_idx % self.config['sync_interval'] == 0:
                    compressed = compress_grad(self.model.grad, self.config)
                    self.comm_queue.put(compressed)
                
                # 接收并融合梯度
                while not self.comm_queue.empty():
                    remote_grad = decompress_grad(self.comm_queue.get())
                    apply_gradient(self.model, remote_grad)
                
                optimizer.step()
                optimizer.zero_grad()

def apply_gradient(model, remote_grad):
    for param, grad in zip(model.parameters(), remote_grad):
        param.grad += grad * 0.3  # 软融合系数
