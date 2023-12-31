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
from drone_detection.msg import drone_sensor, reset
import message_filters
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from pyquaternion import Quaternion
import math
import torch
from Conet.lib.opts import opts
from Conet.lib.models.utils import flip_tensor
import time
from typing_extensions import OrderedDict
import kornia
import threading
from tcp_bridge.msg import ComMessage, Mat2d_33, Mat2d_conf, Mat3d
from Conet.lib.transformation import get_2d_polygon
from Conet.lib.Multi_detection_factory.dla34 import features_extractor, decoder
from Conet.lib.Multi_detection_factory.communication_msg import communication_msg_generator
from Conet.lib.tcp_bridge.tensor2Commsg import msg_process
from Conet.lib.models.decode import ctdet_decode
from Conet.lib.utils.post_process import ctdet_post_process, polygon_nms
from Conet.lib.utils.debugger import Debugger
from numpy.linalg import inv
import os
try:
    from external.nms import soft_nms
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
# temp = sys.stdout
# f = open('screenshot_new.log', 'w')
# sys.stdout = f

class ROS_MultiAgentDetector:
    def __init__(self):
        rospy.init_node('Multi_Detection', anonymous = True)
        self.name_space = rospy.get_namespace().strip('/')
        self.opt = opts().parse()
        self.opt=opts().update_dataset_info_and_set_heads(self.opt)
        self.drone_id = rospy.get_param('/{}/Drone_id'.format(self.name_space))
        self.comm_round = rospy.get_param('/{}/comm_round'.format(self.name_space))
        self.agent_num = rospy.get_param('/{}/agent_num'.format(self.name_space))
        self.image_sub = message_filters.Subscriber("/usb_cam/image_raw".format(self.drone_id), Image)
        self.location_sub = message_filters.Subscriber("/mavros/global_position/local", Odometry)
        self.feature_pub = rospy.Publisher("/{}/ego/feature_data".format(self.name_space), drone_sensor, queue_size=1)
        self.states_list = {'Init':0, 'Start_to_Comm':1,'In_Comm':2}
        self.state = self.states_list['Init']
        self.heads = self.opt.heads
        self.msg_pair = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.lock = threading.Lock()
        self.reset_buffer = OrderedDict()
        self.tcp_trans = msg_process
        self.max_per_image = 100
        self.next_id = 0
        if self.drone_id == 0:
            self.next_id = 1
        self.tcp_pub = rospy.Publisher("/drone_{}_to_drone_{}_sending".format(self.drone_id,self.next_id), ComMessage, queue_size=1)
        self.reset_sub = rospy.Subscriber("/reset_topic", reset, self.state_reset)
        self.height_list = [-1., -0.5, 0., 0.5, 0.75, 1., 1.5, 2., 8.]
        self.camera_intrinsic = np.array([[486.023, 0, 359.066],
                                [0, 486.023, 240.959],
                                [0, 0, 1]])
        self.img_index = 0
        self._valid_ids = [1]
        self.round_id = 0
        self.vis_score_thre = 0.1
        scale_h = 500/500 
        scale_w = 500/500 
        map_scale_h = 1 / scale_h
        map_scale_w = 1 / scale_w
        self.image_size = (int(96/map_scale_h), int(128/map_scale_w))
        self.world_X_left = 200
        self.world_Y_left = 200
        self.worldgrid2worldcoord_mat = np.array([[1, 0, -self.world_X_left], [0, 1, -self.world_Y_left], [0, 0, 1]])
        rospy.Timer(rospy.Duration(0.1), self.Pub_Features)
        self.encoder = features_extractor(pretrained = True, down_ratio =self.opt.down_ratio, feat_shape = self.opt.feat_shape,feat_mode = self.opt.feat_mode,
                                          trans_layer = self.opt.trans_layer).to(self.device)
        self.decoder = decoder(self.heads, self.encoder.base.channels, self.opt.down_ratio, feat_mode = self.opt.feat_mode).to(self.device)
        self.encoder.eval()
        self.decoder.eval()
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.location_sub], 10, 0.5, allow_headerless=False)
        self.ts.registerCallback(self.AlignCallback)
        rospy.spin()
    def AlignCallback(self, Image, Odometry):
        self.lock.acquire()
        self.msg_pair.update({'Image': Image,'Odometry': Odometry})
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
            self.lock.release()
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(Image, "bgr8")
            cv_image = cv2.resize(cv_image, (720, 480))
            # cv_image = cv2.flip(cv_image, 1)
            img_tensor = torch.tensor(cv_image, dtype=torch.float32)
            images=[]
            images.append(img_tensor)
            scaled_images, meta = {}, {}
            for scale in self.opt.test_scales:
                cur_images = []
                print('test_scales: ',scale)
                cur_image, cur_meta =self.encoder.pre_process(cv_image, scale)
                print('image_shape: ',cur_image.shape)
                cur_images.append(cur_image)
                scaled_images[scale] = np.array([np.concatenate(cur_images, axis=0)])
                scaled_images[scale] = torch.from_numpy(scaled_images[scale]).to(torch.float32)
                meta[scale] = cur_meta
                # c = torch.from_numpy(c).to(torch.float32)
                # s = torch.from_numpy(s).to(torch.float32)
                # out_height = np.array([feat_h/(self.opt.map_scale)])
                # out_width = np.array([feat_w/(self.opt.map_scale)])
                # out_height = torch.from_numpy(out_height).to(torch.float32)
                # out_width = torch.from_numpy(out_width).to(torch.float32)
            shift_mats = OrderedDict()
            trans_mats = OrderedDict()
            shift_mats_np = OrderedDict()
            trans_mats_np = OrderedDict()

            orientation = Odometry.pose.pose.orientation
            position = Odometry.pose.pose.position
            roll, pitch, yaw = self.euler_from_quaternion(orientation.x,orientation.y,orientation.z,orientation.w)
            # rotation_1 = Quaternion(self.euler2quaternion(- roll, - yaw, -180)) 
            # rotation_2 = Quaternion(self.euler2quaternion(0, 0,   - pitch))
            # self.rotation = rotation_2.rotation_matrix @ rotation_1.rotation_matrix
            self.local2world = Quaternion(self.euler2quaternion(yaw, pitch, roll))
            # rotation_2 = Quaternion(self.euler2quaternion(180 , 180, 0))
            camera2local = np.diag([1, -1, -1])
            self.rotation = inv(self.local2world.rotation_matrix @ camera2local)
            self.right_to_left = np.diag([1, -1, 1])
            self.im_position = [position.x, position.y, position.z]

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
            
            detections = self.process(preprocessed_Data, Image.header.stamp, Image)
            self.img_transition(trans_mats_np[0.0], shift_mats_np[1], cv_image, detections)   
        except CvBridgeError as e:
            print(e)

    # def process(self,preprocessed_Data, time_stamp, Image):
    #     detections = []
    #     results = []
    #     scale = 1.0
    #     images = preprocessed_Data['images'][scale]
    #     meta = preprocessed_Data['meta'][scale]
    #     if isinstance(meta, list):
    #         updated_meta = []
    #         for cur_meta in meta:
    #             updated_meta.append({k: v.numpy()[0] for k, v in cur_meta.items()})
    #         meta = updated_meta
    #     else:
    #         meta = {k: v.numpy()[0] for k, v in meta.items()}
    #     trans_mats = [preprocessed_Data['trans_mats'], preprocessed_Data['trans_mats_n010'], \
    #                     preprocessed_Data['trans_mats_n005'], preprocessed_Data['trans_mats_p005'],\
    #                     preprocessed_Data['trans_mats_p007'], preprocessed_Data['trans_mats_p010'],\
    #                     preprocessed_Data['trans_mats_p015'], preprocessed_Data['trans_mats_p020'],\
    #                     preprocessed_Data['trans_mats_p080'], preprocessed_Data['trans_mats_withnoise']]
    #     shift_mats = [preprocessed_Data['shift_mats_1'], preprocessed_Data['shift_mats_2'], \
    #                     preprocessed_Data['shift_mats_4'], preprocessed_Data['shift_mats_8'], \
    #                     preprocessed_Data['shift_mats_withnoise']]
    #     # print('origin shift size: ',shift_mats[0].shape)
    #     images = images.to(self.device)
    #     trans_mats = [x.to(self.device) for x in trans_mats]
    #     shift_mats = [x.to(self.device) for x in shift_mats]
    #     global_x, warp_images_list  = self.encoder(images, trans_mats, shift_mats, 1.0)
    #     # print('origin trans_mats: ', trans_mats[0].shape)
    #     output = self.decoder(global_x, self.round_id)[-1]

    #     ##################### Output the Results ########################
    #     dets = self.dets_process(output,shift_mats)
    #     results = []
    #     if isinstance(dets, list):
    #         for cur_dets, cur_meta in zip(dets, meta):
    #             cur_detections = []
    #             cur_results = []
    #             for i in range(len(cur_dets)):
    #                 cur_detections.append(self.post_process(cur_dets[i:i+1], cur_meta, scale))
    #                 cur_results.append(self.merge_outputs([cur_detections[-1]]))
    #             detections.append(cur_detections)
    #             results.append(cur_results)
    #     else:
    #         for i in range(len(dets)):
    #             detections.append(self.post_process(dets[i:i+1], meta, scale))
    #             results.append(self.merge_outputs([detections[-1]]))
    #     detections =  self.Visualization_results(results)
    #     # for anno in detections:
    #     #     if (anno["score"] > 0.03):
    #     #         print(' score: ',anno["score"])
    #     ##################### Output the Results ########################

    #     confidence_map = output['hm'].clone().sigmoid()
        
    #     # print('max: ',confidence_map.max())
    #     # print('results: ',torch.nonzero(communication_mask).shape)
    #     # print('data type: ',type(results))
    #     # for k in results.keys():
    #     #     print('key: ',k,' size: ',results[k].shape)

    #     # self.msg_encoder(global_x, time_stamp, shift_mats, confidence_map, Image, trans_mats)
    #     self.lock.acquire()
    #     # if self.state == self.states_list['Start_to_Comm']:
    #     #     self.state = self.states_list['In_Comm']
    #     self.lock.release()
    #     return detections

    def process(self,image_or_path_or_tensor, time_stamp, Image):
        load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
        merge_time, tot_time = 0, 0
        debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug == 3),
                            theme=self.opt.debugger_theme)
        start_time = time.time()
        pre_processed = False
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(''):
            image = cv2.imread(image_or_path_or_tensor)
        else:
            image = image_or_path_or_tensor['image']
            pre_processed_images = image_or_path_or_tensor
            pre_processed = True

        loaded_time = time.time()
        load_time += (loaded_time - start_time)

        detections = []
        results = []
        comm_rates = []
        for scale in self.opt.test_scales:
            scale_start_time = time.time()
            if not pre_processed:
                images, meta = self.pre_process(image, scale, meta)
            else:
                images = pre_processed_images['images'][scale]
                meta = pre_processed_images['meta'][scale]
                print('images: ',images.shape)
                print('meta: ',meta)
                if isinstance(meta, list):
                    updated_meta = []
                    for cur_meta in meta:
                        updated_meta.append({k: v.numpy()[0] for k, v in cur_meta.items()})
                    meta = updated_meta
                else:
                    meta = {k: v.numpy()[0] for k, v in meta.items()}
                    print('meta after: ',meta)
                trans_mats = [pre_processed_images['trans_mats'], pre_processed_images['trans_mats_n010'], \
                                pre_processed_images['trans_mats_n005'], pre_processed_images['trans_mats_p005'],\
                                pre_processed_images['trans_mats_p007'], pre_processed_images['trans_mats_p010'],\
                                pre_processed_images['trans_mats_p015'], pre_processed_images['trans_mats_p020'],\
                                pre_processed_images['trans_mats_p080'], pre_processed_images['trans_mats_withnoise']]
                shift_mats = [pre_processed_images['shift_mats_1'], pre_processed_images['shift_mats_2'], \
                                pre_processed_images['shift_mats_4'], pre_processed_images['shift_mats_8'], \
                                pre_processed_images['shift_mats_withnoise']]
            images = images.to(self.device)
            # print('trans_mats[0]: ',type(trans_mats[0]))
            trans_mats = [x.to(self.device) for x in trans_mats]
            shift_mats = [x.to(self.device) for x in shift_mats]
            with torch.no_grad():
                global_x, warp_images_list  = self.encoder(images, trans_mats, shift_mats, 1.0)
                output = self.decoder(global_x, self.round_id)[-1]
            
            ##################### Output the Results ########################
            output, dets = self.dets_process(output,shift_mats)
            if self.opt.debug >= 2:
                self.debug(debugger, images, dets, output, scale)
            
            # if self.opt.vis_weight_mats:
            #     self.save_attn_weights(img_idx, images, output)
            if isinstance(dets, list):
                for cur_dets, cur_meta in zip(dets, meta):
                    cur_detections = []
                    cur_results = []
                    for i in range(len(cur_dets)):
                        cur_detections.append(self.post_process(cur_dets[i:i+1], cur_meta, scale))
                        cur_results.append(self.merge_outputs([cur_detections[-1]]))
                    detections.append(cur_detections)
                    results.append(cur_results)
            else:
                for i in range(len(dets)):
                    detections.append(self.post_process(dets[i:i+1], meta, scale))
                    results.append(self.merge_outputs([detections[-1]]))
            torch.cuda.synchronize()
            post_process_time = time.time()
            
        torch.cuda.synchronize()
        end_time = time.time()
        merge_time += end_time - post_process_time
        tot_time += end_time - start_time

        if self.opt.debug >= 1:
            self.show_results(debugger, image, results)

        # return {'results': results, 'tot': tot_time, 'load': load_time,
        #         'pre': pre_time, 'net': net_time, 'dec': dec_time,
        #         'post': post_time, 'merge': merge_time, 'comm_rate': 1}
        rets = results
        detections =  self.Visualization_results(rets)
        # for anno in detections:
        #     if (anno["score"] > 0.03):
        #         print(' score: ',anno["score"])
        ##################### Output the Results ########################

        confidence_map = output['hm'].clone().sigmoid()
        
        # print('max: ',confidence_map.max())
        # print('results: ',torch.nonzero(communication_mask).shape)
        # print('data type: ',type(results))
        # for k in results.keys():
        #     print('key: ',k,' size: ',results[k].shape)

        # self.msg_encoder(global_x, time_stamp, shift_mats, confidence_map, Image, trans_mats)
        self.lock.acquire()
        # if self.state == self.states_list['Start_to_Comm']:
        #     self.state = self.states_list['In_Comm']
        self.lock.release()
        return detections

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
        
    def dets_process(self,output,shift_mats, return_time=False):
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg'] if self.opt.reg_offset else None
        angle = output['angle'] if self.opt.polygon else None # (sin, cos)
        z = output['z'] if 'z' in output else None
        # import ipdb; ipdb.set_trace()
        if self.opt.coord == 'Joint':
            hm_i = output['hm_i'].sigmoid_()
            wh_i = output['wh_i']
            reg_i = output['reg_i'] if self.opt.reg_offset else None
            
        if self.opt.flip_test:
            hm = (hm[0:1] + flip_tensor(hm[1:2])) / 2
            wh = (wh[0:1] + flip_tensor(wh[1:2])) / 2
            reg = reg[0:1] if reg is not None else None
            if self.opt.coord == 'Joint':
                hm_i = (hm_i[0:1] + flip_tensor(hm_i[1:2])) / 2
                wh_i = (wh_i[0:1] + flip_tensor(wh_i[1:2])) / 2
                reg_i = reg_i[0:1] if reg_i is not None else None
        forward_time = time.time()
        if self.opt.coord == 'Local':
            dets = ctdet_decode(hm, wh, map_scale=None, shift_mats=None, reg=reg, angle=None, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        elif self.opt.coord == 'Global':
            dets = ctdet_decode(hm, wh, map_scale=self.opt.map_scale, shift_mats=shift_mats[0], reg=reg, angle=angle, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        else:
            dets_bev = ctdet_decode(hm, wh, map_scale=self.opt.map_scale, shift_mats=shift_mats[0], reg=reg, angle=angle, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
            dets_uav = ctdet_decode(hm_i, wh_i, map_scale=None, shift_mats=None, reg=reg_i, angle=None, cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
            dets = [dets_bev, dets_uav]
        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    
    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32)
            dets[0][j] = dets[0][j].reshape(-1, dets[0][j].shape[-1])
            dets[0][j][:, :(dets[0][j].shape[-1]-1)] /= scale
        return dets[0]
    
    # def get_trans_mat(self,Odometry,z=0):
    #     UAV_height = self.im_position[-1]
    #     im_position = [self.im_position[0],self.im_position[1], UAV_height - z]
    #     im_position = np.array(im_position).reshape((3, 1))
    #     extrinsic_mat = np.hstack((self.rotation, - self.rotation @ im_position))
    #     # reverse_matrix = np.eye(3)
    #     # reverse_matrix[0, 0] = -1
    #     # mat = reverse_matrix @ Quaternion([0.5, -0.5, 0.5, -0.5]).rotation_matrix.T
    #     project_mat = self.camera_intrinsic @ extrinsic_mat
    #     project_mat = np.delete(project_mat, 2, 1) @ self.worldgrid2worldcoord_mat
    #     return project_mat
    
    def get_trans_mat(self,Odometry,z=0):
        UAV_height = self.im_position[-1]
        im_position = [self.im_position[0],self.im_position[1], UAV_height - z]
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
        yaw, _, _ = Quaternion(matrix=self.local2world.rotation_matrix).yaw_pitch_roll
        # yaw = - yaw + math.pi
        x_shift = 60/map_scale_w
        y_shift = 60/map_scale_h
        shift_mat = np.array([[1, 0, -x_shift], [0, 1, -y_shift], [0, 0, 1]])
        rotat_mat = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]) + \
                    np.array([[0, 0, grid_center[0]], [0, 0, grid_center[1]], [0, 0, 0]])
        trans_mat = np.linalg.inv( rotat_mat @ shift_mat)
        return trans_mat, int(Quaternion(matrix=self.rotation).yaw_pitch_roll[0]*180/math.pi)
    
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
        image_g = self.vis_cam(image_g.copy(), detections, shift_mat.copy(), color=(0, 0, 255), vis_thre=self.vis_score_thre)
        cv2.imshow('BEV',image_g)
        cv2.waitKey(3)
    
    def CoordTrans(self, image, project_mat, rotat_mat, mode='L2G'):
        if mode == 'L2G':
            # print(project_mat)
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
    def vis_cam(self, image, annos, rotat_mat, color=(127, 255, 0), vis_thre=-1):
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
                    for index, vaule in enumerate(polygon[:,1]):
                        polygon[index,1] = vaule -15
                    polygon = polygon * 4
                    image = cv2.polylines(image, pts=np.int32([polygon.reshape(-1, 1, 2)]), isClosed=True, color=color, thickness=2)
        return image 

    def trans_message_generation(self, x, shift_mats, require_maps, round_id): #x = (b,agent_num,c,h,w) 其中第一个ego agent，后面的是others，shift也是 [(b,agent_num,3，3)]
        # x: features of ego agent in ego BEV 
        results_dict = {}
        b, c, h, w = x[0].shape
        results = self.decoder(x, round_id)
        results_dict.update(results)
        confidence_maps = results['hm'.format(round_id)].clone().sigmoid()
        confidence_maps = confidence_maps.reshape(b, 1, confidence_maps.shape[-2], confidence_maps.shape[-1])
        require_maps_list = [0,0,0,0]
        ego_request = 1 - confidence_maps.contiguous()
        if round_id > 0:
            require_maps_list[self.opt.trans_layer[0]] = torch.cat([confidence_maps.unsqueeze(1).contiguous(),require_maps],dim=1) # (b, num_agents, 1, h, w)
            require_maps_BEV = self.communication_processor.get_colla_feats(require_maps_list, shift_mats, self.opt.trans_layer) # (b, num_agents, c, h, w) require_maps in BEV maps

        else:
            require_maps_list[self.opt.trans_layer[0]] = confidence_maps.unsqueeze(1).contiguous().expand(-1, self.agent_num, -1, -1, -1).contiguous() # (b, num_agents, 1, h, w)
        val_feats_to_send, _, _, _= self.communication_processor.communication_graph_learning(x[0], confidence_maps, require_maps_BEV[0:], self.agent_num , round_id=0, thre=0.03, sigma=0)
        return val_feats_to_send, ego_request

    def msg_encoder(self, global_x, time_stamp, shift_mats, confidence_map, origin_image, trans_mats):
        data_list=[]
        shift_list = []
        trans_list = []
        channels = []
        for n,v in enumerate(global_x):
            _, c, h, w = global_x[n].size()
            exec('c_dim_{} = MultiArrayDimension(label="channel_{}", size=c, stride=h*w*c)'.format(n,n))
            exec('h_dim_{} = MultiArrayDimension(label="height_{}", size=h, stride=w*h)'.format(n,n))
            exec('w_dim_{} = MultiArrayDimension(label="width_{}", size=w, stride=w)'.format(n,n))
            exec('channels.append(c_dim_{})'.format(n))
            exec('channels.append(h_dim_{})'.format(n))
            exec('channels.append(w_dim_{})'.format(n))
            data_list.extend(global_x[n].to('cpu').detach().numpy().reshape(-1).tolist())
        
        ######################### shift mats #############################################
        shift_mat = shift_mats[0]
        h_shift, w_shift = shift_mat.size()
        shift_list.extend(shift_mat.to('cpu').detach().numpy().reshape(-1).tolist())
        h_dim = MultiArrayDimension(label="height", size=h_shift, stride=h_shift*w_shift)
        w_dim = MultiArrayDimension(label="width",  size=w_shift, stride=w_shift)
        ##################################################################################
        
        ######################### trans mats #############################################
        trans_mat = trans_mats[0]
        h_trans, w_trans = trans_mat.size()
        trans_list.extend(trans_mat.to('cpu').detach().numpy().reshape(-1).tolist())
        h_dim_trans = MultiArrayDimension(label="height", size=h_trans, stride=h_trans*w_trans)
        w_dim_trans = MultiArrayDimension(label="width",  size=w_trans, stride=w_trans)
        ##################################################################################
        
        msg = drone_sensor()
        msg.header.stamp = time_stamp
        msg.drone_id = self.drone_id
        msg.data.layout = MultiArrayLayout(dim=channels, data_offset=0)
        msg.data.data = data_list
        msg.shift_matrix.layout = MultiArrayLayout(dim=[h_dim,w_dim], data_offset=0)
        msg.shift_matrix.data = shift_list
        msg.trans_matrix.layout = MultiArrayLayout(dim=[h_dim_trans,w_dim_trans], data_offset=0)
        msg.trans_matrix.data = trans_list
        msg.image = origin_image
        self.lock.acquire()
        if self.state == self.states_list['In_Comm']:
            print('In Comm stop sending')
            self.lock.release()
            return
        self.lock.release()
        print('send the ego features')
        self.feature_pub.publish(msg)
        
        require_maps = 1 - confidence_map.to('cpu').detach().contiguous()
        require_maps = require_maps.squeeze(0).squeeze(0).contiguous()
        shift_mat_tcp = shift_mat.to('cpu').detach().unsqueeze(0).contiguous()
        features_map_tcp = global_x[0].to('cpu').detach().squeeze(0).contiguous()
        tcp_msg = self.tcp_trans.tensor2Commsg(self.drone_id, self.round_id, shift_mat_tcp, require_maps, features_map_tcp, time_stamp)
        self.lock.acquire()
        if self.state == self.states_list['In_Comm']:
            print('In Comm stop sending')
            self.lock.release()
            return
        self.lock.release()
        print('send the tcp_msg')
        # self.tcp_pub.publish(tcp_msg)
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



    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            if len(self.opt.test_scales) > 1:
                if results[j].shape[-1] > 6:
                    polygon_nms(results[j], 0.5)
                else:
                    soft_nms(results[j], Nt=0.5, method=2)
        scores = np.hstack(
            [results[j][:, -1] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, -1] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    # def Output_results(self, fused_features, )


            # w_dim = MultiArrayDimension(label="width",  size=w, stride=w*c)
            # c_dim = MultiArrayDimension(label="channel", size=c, stride=c)
        #     if n == 0:
        #         global_x_numpy = global_x[n].to('cpu').detach().numpy()
        #     else:
        #         global_x_numpy = np.append(global_x_numpy, global_x[n].to('cpu').detach().numpy(), axis=0)
        # print('size: ',global_x_numpy.size())
    ################# TO DO ###################
    #              pub the msg(confidence_map)                #
    ###########################################
if __name__ == '__main__':
    detector = ROS_MultiAgentDetector()
