#!/usr/bin/env python
# license removed for brevity
from __future__ import absolute_import
from __future__ import division
from torch._C import dtype
import sys
import rospy
import cv2
from std_msgs.msg import String, Float32MultiArray, MultiArrayLayout,MultiArrayDimension,Bool
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from pyquaternion import Quaternion
import math
import torch
import time
from typing_extensions import OrderedDict
import kornia
import threading
from Conet.detector import MultiAgentDetector
from Conet.lib.opts import opts
from Conet.lib.transformation import get_2d_polygon
from Conet.lib.tcp_bridge.Detections2Commsg import post_msg_process
from numpy.linalg import inv
from drone_detection.msg import drone_sensor_postfusion, reset
from post_tcp_bridge.msg import ComMessage, Detection
class ROS_MultiAgentDetector:

    def __init__(self,opt):
        rospy.init_node('ImageProcess', anonymous=True)
        self.name_space = rospy.get_namespace().strip('/')
        self.opt = opt
        self.opt=opts().update_dataset_info_and_set_heads(opt)
        self.detector = MultiAgentDetector(self.opt)
        self.height_list = [-1., -0.5, 0., 0.5, 0.75, 1., 1.5, 2., 8.]
        self.camera_intrinsic = np.array([[486.023, 0, 359.066],
                                [0, 486.023, 240.959],
                                [0, 0, 1]])
        self.img_index = 0
        self._valid_ids = [1]
        self.vis_score_thre = 0.15
        scale_h = 500/500 
        scale_w = 500/500
        self.msg_pair = OrderedDict()
        self.lock = threading.Lock()
        self.agent_num = 1
        self.drone_id = rospy.get_param('/{}/Drone_id'.format(self.name_space))
        self.agent_num = rospy.get_param('/{}/agent_num'.format(self.name_space))
        self.tcp_trans = post_msg_process
        rospy.Timer(rospy.Duration(0.1), self.Pub_Features)
        self.states_list = {'Init':0, 'Start_to_Comm':1,'In_Comm':2}
        self.state = self.states_list['Init']
        map_scale_h = 1 / scale_h
        map_scale_w = 1 / scale_w
        self.image_size = (int(96/map_scale_h), int(192/map_scale_w))
        self.world_X_left = 200
        self.world_Y_left = 200
        self.reset_buffer = OrderedDict()
        self.worldgrid2worldcoord_mat = np.array([[1, 0, -self.world_X_left], [0, 1, -self.world_Y_left], [0, 0, 1]])
        self.image_sub = message_filters.Subscriber("/iris_{}/usb_cam/image_raw".format(self.drone_id), Image)
        self.location_sub = message_filters.Subscriber("/uav{}/mavros/global_position/local".format(self.drone_id), Odometry)
        self.detection_pub = rospy.Publisher("/drone_{}_detection".format(self.drone_id), drone_sensor_postfusion, queue_size=1)
        self.next_id = 0
        if self.drone_id == 0:
            self.next_id = 1
        self.detection_tcp_pub = rospy.Publisher("/drone_{}_to_drone_{}_sending".format(self.drone_id,self.next_id), ComMessage, queue_size=1)
        self.reset_sub = rospy.Subscriber("/reset_topic", reset, self.state_reset)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.location_sub], 10, 0.5, allow_headerless=False)
        self.ts.registerCallback(self.AlignCallback)
        rospy.spin()


    def AlignCallback(self, Image, Odometry):
        self.lock.acquire()
        current_Odometry = Odometry
        if self.drone_id == 0:
            current_x = Odometry.pose.pose.position.x
            current_Odometry.pose.pose.position.x = current_x - 16.5
        else:
            current_x = Odometry.pose.pose.position.x
            current_Odometry.pose.pose.position.x = current_x + 16.5
        self.msg_pair.update({'Image': Image,'Odometry': current_Odometry})
        self.lock.release()
        return

    def Pub_Features(self, event):
        self.lock.acquire()
        if ('Image' not in self.msg_pair.keys() or 'Odometry' not in self.msg_pair.keys()) or self.state == self.states_list['In_Comm']:
            # print('there is no image in buffer')
            # if self.state == self.states_list['In_Comm']:
            #     print('In_Comm stop pub')
            self.lock.release()
            return
        try:
            self.lock.release()
            self.lock.acquire()
            Image = self.msg_pair['Image']
            Odometry = self.msg_pair['Odometry']
            self.msg_pair.clear()
            self.lock.release()
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(Image, "bgr8")
            cv_image = cv2.resize(cv_image, (720, 480))
            # cv_image = cv2.flip(cv_image, 1)
            img_tensor = torch.tensor(cv_image, dtype=torch.float32)
            images=[]
            images.append(img_tensor)
            scaled_images, meta = {}, {}
            for scale in opt.test_scales:
                cur_images = []
                cur_image, cur_meta=self.detector.pre_process(cv_image, scale)
                cur_images.append(cur_image)
                scaled_images[scale] = np.array([np.concatenate(cur_images, axis=0)])
                scaled_images[scale] = torch.from_numpy(scaled_images[scale]).to(torch.float32)
                meta[scale] = cur_meta
            shift_mats = OrderedDict()
            trans_mats = OrderedDict()
            shift_mats_np = OrderedDict()
            trans_mats_np = OrderedDict()

            orientation = Odometry.pose.pose.orientation
            position = Odometry.pose.pose.position
            roll,pitch,yaw =self.euler_from_quaternion(orientation.x ,orientation.y, orientation.z, orientation.w)
            # rotation_1 = Quaternion(self.euler2quaternion(- roll, - yaw, -180)) 
            # rotation_2 = Quaternion(self.euler2quaternion(0, 0,   - pitch))
            # self.rotation = rotation_2.rotation_matrix @ rotation_1.rotation_matrix
            local2world_odm = Quaternion(self.euler2quaternion(yaw, pitch, roll))
            world_odm2world = Quaternion(self.euler2quaternion( 90, 0, 0))
            self.local2world = world_odm2world.rotation_matrix @ local2world_odm.rotation_matrix
            # rotation_2 = Quaternion(self.euler2quaternion(180 , 180, 0))
            camera2local = np.diag([1, -1, -1])
            self.rotation = inv( world_odm2world.rotation_matrix @ local2world_odm.rotation_matrix @ camera2local)
            self.right_to_left = np.diag([1, -1, 1])
            self.im_position = [position.x, position.y, position.z]

            # print('drone_{}'.format(self.drone_id), ' im_position: ',self.im_position)
            for scale in [1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16, 32, 64]:
                cur_shift_mat, _ = self.get_crop_shift_mat(map_scale_w=scale, map_scale_h=scale, \
                                                            world_X_left=self.world_X_left, world_Y_left=self.world_Y_left) 
                shift_mats_np[scale] = cur_shift_mat
                shift_mats[scale] = torch.from_numpy(cur_shift_mat).to(torch.float32)
            for height in self.height_list :
                cur_trans_mat = self.get_trans_mat(Odometry = Odometry, z = height)
                trans_mats_np[height] = cur_trans_mat
                trans_mats[height] = torch.from_numpy(cur_trans_mat).to(torch.float32)
            preprocessed_Data = {'images': scaled_images, 'image': images, 'meta': meta, \
                            'trans_mats': trans_mats[0.0], 'trans_mats_n010': trans_mats[-1.0], 'trans_mats_n005': trans_mats[-0.5], 'trans_mats_p005': trans_mats[0.5],\
                            'trans_mats_p007': trans_mats[0.75], 'trans_mats_p010': trans_mats[1.0], 'trans_mats_p015': trans_mats[1.5], 'trans_mats_p020': trans_mats[2.0],\
                            'trans_mats_p080': trans_mats[8.0], 
                            'shift_mats_1': shift_mats[1], 'shift_mats_2': shift_mats[2], 'shift_mats_4': shift_mats[4], 'shift_mats_8': shift_mats[8],
                            'trans_mats_withnoise': trans_mats[8.0], 'shift_mats_withnoise': shift_mats[8]}
            ret = self.detector.run(preprocessed_Data, self.img_index)
            self.img_index+=1
            # cv2.imshow('BEV', cv_image)
            # cv2.waitKey(3)
            rets = ret['results']
            detections = self.Visualization_results(rets)
            self.pack_ego_detection(detections,shift_mats[1],trans_mats[0.0], Image.header.stamp, Image)
            # self.img_transition(trans_mats_np[0.0], shift_mats_np[1], cv_image, detections)
            # print('result: ',rets)
        except CvBridgeError as e:
            print(e)

    def pack_ego_detection (self, detections, shift_mats, trans_mats, time_stamp, origin_image):
        detection_size = len(detections)
        data_len = 10
        data_matrix=torch.zeros(detection_size, data_len)
        shift_list = []
        trans_list = []
        detection_list = []
        for n, v in enumerate(detections):
            data_matrix[n,0]  = torch.tensor(v["category_id"], dtype=torch.float32)
            data_matrix[n,1:] = torch.tensor(np.array(v["bbox"]),dtype=torch.float32)
        # print('origin detection: ',detections)
        # print('tensor_matrix: ',data_matrix)
        
        ######################### data mats #############################################
        h_detection, w_detection = data_matrix.size()
        detection_list.extend(data_matrix.to('cpu').detach().numpy().reshape(-1).tolist())
        h_dim_detection = MultiArrayDimension(label="height", size=h_detection, stride=h_detection*w_detection)
        w_dim_detection = MultiArrayDimension(label="width",  size=w_detection, stride=w_detection)
        ######################### shift mats #############################################
        shift_mat = shift_mats
        h_shift, w_shift = shift_mat.size()
        shift_list.extend(shift_mat.to('cpu').detach().numpy().reshape(-1).tolist())
        h_dim = MultiArrayDimension(label="height", size=h_shift, stride=h_shift*w_shift)
        w_dim = MultiArrayDimension(label="width",  size=w_shift, stride=w_shift)
        ##################################################################################
        
        ######################### trans mats #############################################
        trans_mat = trans_mats
        h_trans, w_trans = trans_mat.size()
        trans_list.extend(trans_mat.to('cpu').detach().numpy().reshape(-1).tolist())
        h_dim_trans = MultiArrayDimension(label="height", size=h_trans, stride=h_trans*w_trans)
        w_dim_trans = MultiArrayDimension(label="width",  size=w_trans, stride=w_trans)
        ##################################################################################
        
        msg = drone_sensor_postfusion()
        msg.header.stamp = time_stamp
        msg.drone_id = self.drone_id
        msg.detections.layout = MultiArrayLayout(dim=[h_dim_detection, w_dim_detection], data_offset=0)
        msg.detections.data = detection_list
        msg.shift_matrix.layout = MultiArrayLayout(dim=[h_dim,w_dim], data_offset=0)
        msg.shift_matrix.data = shift_list
        msg.trans_matrix.layout = MultiArrayLayout(dim=[h_dim_trans,w_dim_trans], data_offset=0)
        msg.trans_matrix.data = trans_list
        msg.image = origin_image
        # self.lock.acquire()
        # if self.state == self.states_list['In_Comm']:
        #     print('In Comm stop sending')
        #     self.lock.release()
        #     return
        # self.lock.release()
        self.detection_pub.publish(msg)
        shift_mat_tcp = shift_mat.to('cpu').detach().contiguous()
        tcp_msg = self.tcp_trans.Detections2Commsg(time_stamp, detections, self.drone_id, 0, shift_mat_tcp)
        print('origin data: ',self.drone_id ,' detection: ',detections,' shift_mat_tcp: ',shift_mat_tcp)
        self.lock.acquire()
        if self.state == self.states_list['In_Comm']:
            print('In Comm stop sending')
            self.lock.release()
            return
        self.lock.release()
        print('send the tcp_msg')
        self.detection_tcp_pub.publish(tcp_msg)

    def get_trans_mat(self,Odometry,z=0):
        UAV_height = self.im_position[-1]
        im_position = [self.im_position[0],self.im_position[1], UAV_height - z]
        print('im_position: ',im_position)
        im_position = np.array(im_position).reshape((3, 1))
        extrinsic_mat = np.hstack((self.rotation, - self.rotation @ im_position))
        # reverse_matrix = np.eye(3)
        # reverse_matrix[0, 0] = -1
        # mat = reverse_matrix @ Quaternion([0.5, -0.5, 0.5, -0.5]).rotation_matrix.T
        project_mat = self.camera_intrinsic @ self.right_to_left @ extrinsic_mat
        project_mat = np.delete(project_mat, 2, 1) @ self.worldgrid2worldcoord_mat
        return project_mat
    
    def get_crop_shift_mat(self, map_scale_w=1, map_scale_h=1, world_X_left=200, world_Y_left=200):
        im_position = [self.im_position[0], self.im_position[1], 1.]
        world_mat = np.array([[1/map_scale_w, 0, 0], [0, 1/map_scale_h, 0], [0, 0, 1]]) @ \
                np.array([[1, 0, world_X_left], [0, 1, world_Y_left], [0, 0, 1]])
        grid_center = world_mat @ im_position
        # print('grid_center: ',grid_center,'scale: ',map_scale_w)
        yaw, _, _ = Quaternion(matrix=self.rotation).yaw_pitch_roll
        yaw = - yaw 
        # yaw = - yaw
        x_shift = 60/map_scale_w
        y_shift = 60/map_scale_h
        shift_mat = np.array([[1, 0, -x_shift], [0, 1, -y_shift], [0, 0, 1]])
        rotat_mat = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]) + \
                    np.array([[0, 0, grid_center[0]], [0, 0, grid_center[1]], [0, 0, 0]])
        trans_mat = np.linalg.inv( rotat_mat @ shift_mat)
        return trans_mat, int(Quaternion(matrix=self.rotation).yaw_pitch_roll[0]*180/math.pi)
    
    def Visualization_results(self,detection_results):
        detections = []
        for detection_result in detection_results:
            category_id = self._valid_ids[0]
            for bbox in detection_result[category_id]:
                if len(bbox) > 5:
                    bbox_out = [float("{:.2f}".format(bbox[i])) for i in range(len(bbox-1))]
                    score = bbox[-1]
                else:
                    bbox[2] -= bbox[0]
                    bbox[3] -= bbox[1]
                    score = bbox[4]
                    bbox_out = [float("{:.2f}".format(bbox[i])) for i in range(4)]
                detection = {
                    "image_id": int(self.img_index),
                    "category_id": int(category_id),
                    "bbox": bbox_out,
                    "score": float("{:.2f}".format(score))
                }
                if detection["score"] > self.vis_score_thre:
                    detections.append(detection)
        return detections
    
    def euler2quaternion(self, yaw, pitch, roll):
        cy = np.cos(yaw * 0.5 * np.pi / 180.0)
        sy = np.sin(yaw * 0.5 * np.pi / 180.0)
        cp = np.cos(pitch * 0.5 * np.pi / 180.0)
        sp = np.sin(pitch * 0.5 * np.pi / 180.0)
        cr = np.cos(roll * 0.5 * np.pi / 180.0)
        sr = np.sin(roll * 0.5 * np.pi / 180.0)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return w, x, y, z
    def euler_from_quaternion(self, x, y, z, w):

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
    
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
    
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z    
    def img_transition(self,trans_mat_input,shift_mats_input,image,detections):
        # image_u = cv2.resize(image, (720, 480))
        trans_mat = trans_mat_input.copy()
        shift_mat = shift_mats_input.copy()
        # translation = np.zeros(3,1)
        image_g = self.CoordTrans(image.copy(), trans_mat.copy(), shift_mat.copy())
        image_g = self.vis_cam(image_g.copy(), detections, color=(0, 0, 255), vis_thre=self.vis_score_thre)
        cv2.imshow('BEV',image_g)
        cv2.waitKey(3)
    
    def CoordTrans(self, image, project_mat, rotat_mat, mode='L2G'):
        if mode == 'L2G':
            print(project_mat)
            trans_mat = np.linalg.inv(project_mat)
        else:
            trans_mat = project_mat
        
        feat_mat = np.array(np.diag([4, 4, 1]), dtype=np.float32)
        #tans is from RV to world grid
        #rotat_mat is from grid world  to camera world
        trans_mat = feat_mat @ rotat_mat @ trans_mat
        data = kornia.image_to_tensor(image, keepdim=False)
        data_warp = kornia.warp_perspective(data.float(),
                                            torch.tensor(trans_mat).repeat([1, 1, 1]).float(),
                                            dsize=(self.image_size[0]*4, self.image_size[1]*4))
        img_warp = kornia.tensor_to_image(data_warp.byte())
        return img_warp
    def vis_cam(self, image, annos, color=(127, 255, 0), vis_thre=-1):
    # image = np.ones_like(image) * 255
    # image = np.array(image * 0.85, dtype=np.int32)
    # alpha = np.ones([image.shape[0], image.shape[1], 1]) * 100
    # image = np.concatenate([image, alpha], axis=-1)
    # color = (255, 255, 255)
        for anno in annos:
            if (anno["score"] > vis_thre):
                bbox = anno["bbox"]
                if len(bbox) == 4:
                    bbox = [x*4 for x in bbox]
                    x, y, w, h = bbox
                    image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
                else:
                    polygon = np.array(get_2d_polygon(np.array(bbox[:8]).reshape([4,2]).T)).reshape([4,2])
                    # for index, vaule in enumerate(polygon[:,1]):
                    #     polygon[index,1] = vaule -15
                    polygon = polygon * 4
                    image = cv2.polylines(image, pts=np.int32([polygon.reshape(-1, 1, 2)]), isClosed=True, color=color, thickness=2)
        return image

    def state_reset(self,msg):
        print ('get drone: ', msg.drone_id,' reset require: ', msg.reset)
        self.lock.acquire()
        self.reset_buffer.update({'drone_{}'.format(msg.drone_id):msg.reset})
        self.lock.release()
        if len(self.reset_buffer) == self.agent_num:
            if msg.reset :
                print('send new image')
                self.lock.acquire()
                self.state = self.states_list['Start_to_Comm']
                self.lock.release()
            else:
                print('stop pub image')
                self.lock.acquire()
                self.state = self.states_list['In_Comm']
                self.lock.release()
            self.lock.acquire()    
            self.reset_buffer.clear()
            self.lock.release()
        return       
if __name__ == '__main__':
    opt = opts().parse()
    ROS_MultiAgentDetector(opt)
