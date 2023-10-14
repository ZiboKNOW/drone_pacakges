#! /usr/bin/env python

import rospy
from std_msgs.msg import Header
from post_tcp_bridge.msg import ComMessage, Detection
import numpy as np
import torch
class post_msg_process: 
    def __init__(self) -> None:
        pass
    def Detections2Commsg(time_stamp ,detections : list, drone_id, round_id, shift_mat : torch.Tensor):
        msg = ComMessage()
        msg.header = Header()
        msg.drone_id = drone_id
        msg.round_id = round_id
        msg.header.stamp = time_stamp
        mat = [x[i] for x in shift_mat.numpy().tolist() for i in range(3)]
        for i in range(9):
            msg.shift_mat[i] = mat[i]
        
        for i in range(len(detections)):
            tmp = Detection()
            tmp.category_id = detections[i]["category_id"]
            print(detections[i]["bbox"])
            tmp.bbox = detections[i]["bbox"]
            msg.detections.append(tmp)
        
        return msg

    def Commsg2Detections(msg : ComMessage):
        drone_id = msg.drone_id
        round_id = msg.round_id

        shift_mat = torch.Tensor(msg.shift_mat)
        shift_mat = shift_mat.reshape(3, 3)

        detections = []
        for detection in msg.detections:
            tmp = {"category_id": detection.category_id, "bbox": detection.bbox}
            detections.append(tmp)
        

        return drone_id, round_id, shift_mat, detections

tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) #shift matrix
print(tensor1.shape)
# tensor2 = torch.tensor([[4, 5, 6], [4, 5, 6]]) #confidence map
# tensor3 = torch.randn(3,3,3) #features_map
# print(tensor1.shape, tensor2.shape, tensor3.shape)
# # print(tensor3.numpy().tolist())

# detections = [{"category_id" : 1, "bbox": [0, 1, 1, 1]}, {"category_id" : 2, "bbox": [4, 5, 6, 7]}]


# res = Commsg2Detections(msg)
# print(res)


