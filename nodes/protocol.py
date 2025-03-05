import numpy as np
import zlib

def compress_grad(grad_tensor, config):
    """梯度压缩协议"""
    grad_np = grad_tensor.cpu().numpy()
    flattened = grad_np.flatten()
    
    # Top-K稀疏化
    k = int(len(flattened) * config['sparse_ratio'])
    indices = np.argpartition(np.abs(flattened), -k)[-k:]
    sparse_grad = flattened[indices]
    
    # 整数量化
    max_val = np.max(np.abs(sparse_grad))
    quantized = np.round(127 * sparse_grad / max_val).astype(np.int8)
    
    # 索引压缩
    bitmap = np.zeros(len(flattened), dtype=bool)
    bitmap[indices] = True
    compressed_idx = zlib.compress(bitmap.tobytes(), level=3)
    
    return {
        'indices': compressed_idx,
        'values': quantized,
        'scale': max_val
    }

def decompress_grad(compressed):
    """梯度解压协议""" 
    # 解压索引
    bitmap = np.frombuffer(zlib.decompress(compressed['indices']), dtype=bool)
    indices = np.where(bitmap)[0]
    
    # 反量化
    values = compressed['values'].astype(np.float32) 
    values *= compressed['scale'] / 127
    
    # 重构梯度张量
    grad_full = np.zeros(bitmap.shape, dtype=np.float32)
    grad_full[indices] = values
    return grad_full.reshape(compressed['original_shape'])
