import os
import sys

# 环境检测与路径配置
IS_KAGGLE = "kaggle" in os.getcwd().lower()
PROJECT_ROOT = "/kaggle/working/distributed_ai" if IS_KAGGLE else os.path.dirname(__file__)
sys.path.append(PROJECT_ROOT)

# 动态导入模块
from nodes.node import VirtualNode
from utils.monitor import ResourceMonitor

from multiprocessing import Queue
from nodes.node import VirtualNode

if __name__ == "__main__":
    # 初始化通信队列
    comm_queue = Queue(maxsize=100)
    
    # 启动4个节点进程
    processes = []
    for i in range(4):
        p = VirtualNode(node_id=i, comm_queue=comm_queue, 
                       config_path='configs/comm_config.yaml')
        p.start()
        processes.append(p)
    
    # 启动资源监控
    from utils.monitor import ResourceMonitor
    monitor = ResourceMonitor(port=8000)
    monitor.collect_metrics()
    
    # 等待训练完成
    [p.join() for p in processes]
