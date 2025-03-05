import psutil
import time
from prometheus_client import start_http_server, Gauge

class ResourceMonitor:
    def __init__(self, port=8000):
        self.cpu_usage = Gauge('cpu_usage', 'CPU使用率')
        self.mem_usage = Gauge('mem_usage', '内存使用率')
        self.gpu_usage = Gauge('gpu_usage', 'GPU使用率')
        start_http_server(port)
        
    def collect_metrics(self):
        while True:
            # CPU监控
            self.cpu_usage.set(psutil.cpu_percent())
            
            # 内存监控
            mem = psutil.virtual_memory()
            self.mem_usage.set(mem.percent)
            
            # GPU监控（需NVIDIA库）
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.gpu_usage.set(util.gpu)
            except ImportError:
                pass
            
            time.sleep(5)
