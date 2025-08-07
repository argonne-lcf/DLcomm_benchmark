 

import torch

def dummy_mxm_compute(device, dtype=None, size=1024):
    dtype = torch.float32
    
    A = torch.randn(size, size, dtype=dtype, device=device)
    B = torch.randn(size, size, dtype=dtype, device=device)
    C = torch.matmul(A, B)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'xpu':
        torch.xpu.synchronize()
    
    del A, B, C

if __name__ == "__main__":
    # Test with XPU device (Intel GPU)
    if torch.xpu.is_available():
        device = torch.device('xpu:0')
        print(f"Running 512x512 matrix multiplication on {device}")
    else:
        device = torch.device('cpu')
        print(f"XPU not available, running on {device}")
    
    # Run multiple iterations for better timing data
    for i in range(10):
        dummy_mxm_compute(device, size=21600)
    
    print("Matrix multiplication test completed")