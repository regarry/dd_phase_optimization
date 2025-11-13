import torch
import os
import numpy as np

print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.get_device_name():", torch.cuda.get_device_name())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
arr = np.array([[1, 2, 3], [4, 5, 6]])
tensor = torch.from_numpy(arr)
tensor.to(device='cuda:0')