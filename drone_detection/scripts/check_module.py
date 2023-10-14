import torch
from drone_detection.msg import drone_sensor
matrix = torch.random(1,3,3,3)
matrix_2 = matrix[:,1].unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
print(matrix)
print(matrix_2)


