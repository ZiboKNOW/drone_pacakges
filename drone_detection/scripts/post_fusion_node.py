from __future__ import absolute_import
from __future__ import division
from torch._C import dtype
import sys
import rospy
import cv2
from std_msgs.msg import String, Float32MultiArray, MultiArrayLayout, MultiArrayDimension, Header
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import message_filters
from drone_detection.msg import drone_sensor, reset, drone_sensor_postfusion
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from pyquaternion import Quaternion
import math
import torch
from torch import nn, sigmoid
import time
from typing_extensions import OrderedDict
from Conet.lib.models.decode import ctdet_decode
import kornia
import threading
from numpy.linalg import inv
from Conet.lib.Multi_detection_factory.communication_msg import communication_msg_generator
from Conet.lib.Multi_detection_factory.dla34 import decoder
from Conet.lib.tcp_bridge.Detections2Commsg import post_msg_process
from post_tcp_bridge.msg import ComMessage, Detection
from Conet.lib.utils.post_process import ctdet_post_process, polygon_nms
from Conet.lib.transformation import get_2d_polygon
try:
    from external.nms import soft_nms
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
# temp = sys.stdout
# f = open('fusion_node.log', 'w')
# sys.stdout = f
class features_fusion():
    def __init__(self):
        rospy.init_node('Features_Fusion', anonymous = True)
        self.name_space = rospy.get_namespace().strip('/')
        self.drone_id = rospy.get_param('/{}/Drone_id'.format(self.name_space))
        self.agent_num = rospy.get_param('/{}/agent_num'.format(self.name_space))
        # self.name_space = rospy.get_namespace().strip('/')
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.decoder_sub = rospy.Subscriber("/drone_{}_detection".format(self.drone_id), drone_sensor_postfusion, self.ego_msg_decoder)
        self.tcp_trans = post_msg_process
        self.reset_pub = rospy.Publisher("/reset_topic", reset, queue_size= 1)
        # self.final_pub = rospy.Publisher("/drone_{}_final_result".format(self.drone_id), drone_sensor, queue_size= 1)
        self.lock = threading.Lock()
        # self.features_map = OrderedDict()
        # self.shift_map = OrderedDict()
        self.time_gap_threshold = 0.2
        self.round_id = int(0)
        self.fusion_buffer = OrderedDict()
        self.lock = threading.Lock()
        self.buffer_lock = False
        self.origin_image = 0
        self.origin_trans_mat = np.zeros((3,3))
        self.image_size = (int(96), int(192))
        self.vis_score_thre = 0.2
        self.max_per_image = 100
        self._valid_ids = [1]
        self.img_index = 0
        self.in_fusion = False
        self.next_id = 0
        if self.drone_id == 0:
            self.next_id = 1
        self.tcp_pub = rospy.Publisher("/drone_{}_to_drone_{}_sending".format(self.drone_id,self.next_id), ComMessage, queue_size=1)
        self.detection_sub = rospy.Subscriber('/drone_{}_recive'.format(self.drone_id), ComMessage, self.drones_msg_decoder)
        # self.decoder = decoder(self.heads, self.channels, self.down_ratio).to(self.device)
        rospy.Timer(rospy.Duration(0.1), self.Check_Buffer)
        self.round_start_time = time.time()
        rospy.spin()

    def send_reset(self, send_new_msg):
        msg = reset()
        msg.drone_id.data = self.drone_id
        msg.reset = send_new_msg
        msg_Header = Header()
        msg_Header.stamp = rospy.get_rostime()
        msg.header = msg_Header
        if send_new_msg:
            self.round_id = 0
            self.lock.acquire()
            self.fusion_buffer.clear()
            self.lock.release()
        self.reset_pub.publish(msg)

    def Check_Buffer(self,event):
        self.lock.acquire()
        if len(self.fusion_buffer) == self.agent_num:
            for k in self.fusion_buffer.keys():
                if 'round_{}'.format(self.round_id) not in k:
                    print('Error: Wrong msg in buffer.')
                    self.send_reset(True)
                    self.lock.release()
                    return
            self.lock.release()
            self.post_fusion()  

        else:
            self.lock.release()
            return
    def get_meta(self, feat_shape, map_scale):
        feat_w, feat_h = feat_shape
        c = np.array([feat_w/(2*map_scale), feat_h/(2*map_scale)])
        s = np.array([feat_w/(map_scale), feat_h/(map_scale)])
        c = torch.from_numpy(c).to(torch.float32)
        s = torch.from_numpy(s).to(torch.float32)
        out_height = np.array([feat_h/(map_scale)])
        out_width = np.array([feat_w/(map_scale)])
        out_height = torch.from_numpy(out_height).to(torch.float32)
        out_width = torch.from_numpy(out_width).to(torch.float32)
        meta = {'c': c, 's': s,
                'out_height': out_height,
                'out_width': out_width}            
        return meta

    def drones_msg_decoder(self, ComMessage):
        drone_id, agent_round_id, shift_mat, detections = self.tcp_trans.Commsg2Detections(ComMessage)
        print('drone_id: ',drone_id, ' round_id: ',agent_round_id,' shift_mat: ',shift_mat,' detections: ',detections)
        update_dict = False
        if self.round_id == 0:
            if agent_round_id == self.round_id and not self.in_fusion:
                update_dict = True
            elif agent_round_id > self.round_id and self.in_fusion:
                update_dict = True
            elif agent_round_id == self.round_id and self.in_fusion:
                print('In fusion: stop pub image')
                self.send_reset(False)
                return
            else:
                print('error: reset the iamge')
                self.send_reset(True)
                return
        else:
            if self.round_id <= agent_round_id:
                update_dict = True
            elif self.in_fusion :
                print('In fusion: stop pub image')
                self.send_reset(False)
            else:
                print('Error: late msg')
                self.send_reset(True)             
        if update_dict:
            drone_msg_data= OrderedDict()
            drone_msg_data['detecion'] = detections #[b, c, h, w]
            drone_msg_data['shift_mat'] = shift_mat.numpy()
            drone_msg_data['round_id'] = agent_round_id
            drone_msg_data['header'] = ComMessage.header
            self.lock.acquire()
            self.fusion_buffer.update({'drone_{}_round_{}_msg'.format(drone_id, agent_round_id) : drone_msg_data})
            self.lock.release()
            # print('decoded shift_mat: ',drone_msg_data['shift_mat'])
            # print('ego shape ','features_map: ',drone_msg_data['features_map'][0].shape,' shift_mat: ',drone_msg_data['shift_mat'][0].shape)
        else:
            return

    def post_fusion(self):
        print('start fusion')
        fusion_start_time = time.time()
        if self.round_id == 0:
            self.round_start_time = time.time()
        self.in_fusion = True
        time_stamp_list = []
        self.lock.acquire()
        for k in self.fusion_buffer.keys():
            time_stamp_list.append(self.fusion_buffer[k]['header'].stamp.secs)
        self.lock.release()
        if self.round_id == 0 and max(time_stamp_list) - min(time_stamp_list) > self.time_gap_threshold:
            print('Error: too large stamp gap(secs) ',max(time_stamp_list) - min(time_stamp_list))
            self.send_reset(True)
            self.in_fusion = False
            return
        self.lock.acquire()
        target_shift_mat = self.fusion_buffer['drone_{}_round_{}_msg'.format(self.drone_id,self.round_id)]['shift_mat']
        co_polygon_list = []
        origin_detections = self.fusion_buffer['drone_{}_round_{}_msg'.format(self.drone_id,self.round_id)]['detecion']
        origin_polygon = self.detections2polygon(origin_detections)
        for k in self.fusion_buffer:
            if 'drone_{}'.format(self.drone_id) not in k:
                detections = self.fusion_buffer[k]['detecion']
                origin_shift = self.fusion_buffer[k]['shift_mat']
                current_polygons = self.polygon_trans(detections, target_shift_mat, origin_shift)
                co_polygon_list += current_polygons
        self.lock.release()
        if self.round_id == 0:
            bridge = CvBridge()
            self.lock.acquire()
            origin_Image = self.fusion_buffer['drone_{}_round_{}_msg'.format(self.drone_id,self.round_id)]['Image']
            self.origin_trans_mat = self.fusion_buffer['drone_{}_round_{}_msg'.format(self.drone_id,self.round_id)]['trans_mat']
            self.lock.release()
            cv_image = bridge.imgmsg_to_cv2(origin_Image, "bgr8")
            cv_image = cv2.resize(cv_image, (720, 480))
            # cv_image = cv2.flip(cv_image, 1)
            self.origin_image = cv_image
        if self.round_id >= 0:
            print("End fusion, require for new image")
            # output = self.decoder(fused_feature_list, self.round_id)[-1]
            # self.show_results(output,shift_mat_list[0][0, self.drone_id])
            # round_end_time = time.time()
            # print('round time: ',round_end_time - self.round_start_time,'s')
            self.in_fusion = False
            self.post_fusion_result(self.origin_trans_mat, target_shift_mat, self.origin_image, origin_polygon, co_polygon_list)
            self.send_reset(True)
            return
        
        # clear the buffer
        self.lock.acquire()
        for k in list(self.fusion_buffer.keys()):
            if 'round_'.format(self.round_id) in k:
                del self.fusion_buffer[k]
        self.round_id +=1
        self.lock.release()

        return
                
    def detections2polygon(self,detections):
        polygon_list = []
        for anno in detections:
            bbox = anno["bbox"]
            polygon = np.array(get_2d_polygon(np.array(bbox[:8]).reshape([4,2]).T)).reshape([4,2])
            for index, vaule in enumerate(polygon[:,1]):
                polygon[index,1] = vaule - 15
            polygon = polygon * 4
            polygon_list.append(polygon)
        return polygon_list                  
    def polygon_trans(self, detections, target_shift_mat, origin_shift):
        polygon_list =[]
        for anno in detections:
            bbox = anno["bbox"]
            if len(bbox) == 4:
                bbox = [x*4 for x in bbox]
                x, y, w, h = bbox
            else:
                polygon = np.array(get_2d_polygon(np.array(bbox[:8]).reshape([4,2]).T)).reshape([4,2])
                # polygon = polygon * 4
                for index, vaule in enumerate(polygon[:,1]):
                    polygon[index,1] = vaule - 15
                point_index, point_size = polygon.shape
                shift_polygons = np.zeros((point_index,point_size))
                print('{}_origin_shift: '.format(self.next_id), origin_shift)
                print('target_shift_mat: ', target_shift_mat)
                for i in range(point_index):
                    current_point = np.append(polygon[i,:], 1)
                    print('current_point: ',polygon[i,:] * 4)
                    trans_point =  target_shift_mat @ inv(origin_shift) @ current_point 
                    trans_point = trans_point / trans_point[-1]
                    trans_point = trans_point[:-1].copy() * 4
                    print('shift_point: ',trans_point)
                    shift_polygons[i,:] = trans_point
                polygon_list.append(shift_polygons)
        
        return polygon_list
      
    # def pack_ego_msg(self, fused_feature,shift_mat):
    #     # shift_mat
    #     drone_msg_data= OrderedDict()
    #     drone_msg_data['features_map'] = fused_feature #[b, c, h, w]
    #     print('packing ego msg: ',len(fused_feature))
    #     drone_msg_data['shift_mat'] = shift_mat
    #     # print('shift_size: ',shift_mat[0].shape ,'len: ',len(shift_mat))
    #     for i in self.trans_layer:
    #         drone_msg_data['shift_mat'][i] = shift_mat[i][0,self.drone_id]
    #         # print('drone_msg_data shift_mat: ', drone_msg_data['shift_mat'][i].shape)
    #     drone_msg_data['round_id'] = self.round_id
    #     msg_header = Header()
    #     msg_header.stamp = rospy.Time.now()
    #     drone_msg_data['header'] = msg_header
    #     return drone_msg_data

    def ego_msg_decoder(self, data):
        detections = []
        layout_detections = data.detections.layout
        detections_msg = list(data.detections.data)
        h_detections = layout_detections.dim[0].size   
        w_detections = layout_detections.dim[1].size
        origin_detections_matrix = np.array(detections_msg, dtype=np.float32).reshape(h_detections,w_detections)
        for i in range(h_detections):
            category_id = origin_detections_matrix[i,0]
            bbox = origin_detections_matrix[i,1:].tolist()
            score = origin_detections_matrix[i, -1]
            detection = {
                "category_id": int(category_id),
                "bbox": bbox,
                "score": float("{:.2f}".format(score))
            }
            detections.append(detection)
        # print('decoded detections: ',detections)
        layout_shift = data.shift_matrix.layout
        shift_msg = list(data.shift_matrix.data)
        h_shift = layout_shift.dim[0].size   
        w_shift = layout_shift.dim[1].size
        # origin_shift_matrix = torch.tensor(np.array(shift_msg, dtype=np.float32).reshape(h_shift,w_shift))
        origin_shift_matrix = np.array(shift_msg, dtype=np.float32).reshape(h_shift,w_shift)
        layout_trans = data.trans_matrix.layout
        trans_msg = list(data.trans_matrix.data)
        h_trans = layout_trans.dim[0].size   
        w_trans = layout_trans.dim[1].size
        origin_trans_matrix = np.array(trans_msg, dtype=np.float32).reshape(h_trans,w_trans)
        # for v in feature_list:
        #     print('feature_size: ',v.size())
        
        drone_msg_data= OrderedDict()
        drone_msg_data['detecion'] = detections #[b, c, h, w]
        drone_msg_data['shift_mat'] = origin_shift_matrix
        drone_msg_data['trans_mat'] = origin_trans_matrix
        drone_msg_data['header'] = data.header
        drone_msg_data['Image'] = data.image
        self.lock.acquire()
        self.fusion_buffer.update({'drone_{}_round_{}_msg'.format(self.drone_id, self.round_id):drone_msg_data})
        self.lock.release()
        return

    def trans_message_generation(self, fused_features, shift_mats, agent_require_maps, round_id): 
        results_dict = {}
        b, c, h, w = fused_features[0].shape
        results = self.decoder(fused_features, round_id)[-1]
        results_dict.update(results)
        confidence_maps = results['hm'].clone().sigmoid()
        confidence_maps = confidence_maps.reshape(b, 1, confidence_maps.shape[-2], confidence_maps.shape[-1])
        # print('confidence_maps size: ',confidence_maps.shape) #[1, 1, 96, 192]
        require_maps_list = [0,0,0,0]
        ego_request = 1 - confidence_maps.contiguous()  #[1, 1, 96, 192]
        require_maps = agent_require_maps
        require_maps = require_maps.unsqueeze(2).contiguous() #[1,2,1,96, 192]
        require_maps[0,self.drone_id,0] = ego_request[0,0]
        require_maps_list[self.trans_layer[0]] = require_maps # (b, num_agents, h, w)
        require_maps_BEV = self.communication_module.get_colla_feats(require_maps_list, shift_mats, self.trans_layer) # (b, num_agents, c, h, w) require_maps in BEV maps
        # else:
        #     require_maps_list[self.trans_layer[0]] = confidence_maps.unsqueeze(1).contiguous().expand(-1, self.agent_num, -1, -1, -1).contiguous() # (b, num_agents, 1, h, w)
        # print('require_maps_BEV size: ',require_maps_BEV[0].shape, 'len: ',len(require_maps_BEV)) # [1, 2, 1, 96, 192]
        val_feats_to_send, _, _= self.communication_module.communication_graph_learning(fused_features[0], confidence_maps, require_maps_BEV[0], self.agent_num , round_id , thre=0.03, sigma=0)
        # print('val_feats_to_send: ', val_feats_to_send.shape, 'ego_request: ',ego_request.shape)
        return val_feats_to_send, ego_request

    def send_tcp_msg(self, val_feats_to_send, ego_request,next_id):
        self.lock.acquire()
        ego_msg = self.fusion_buffer['drone_{}_round_{}_msg'.format(self.drone_id,self.round_id)]
        self.lock.release()
        tcp_msg = self.tcp_trans.tensor2Commsg(self.drone_id, self.round_id, ego_msg['shift_mat'][0].to('cpu').detach().unsqueeze(0).contiguous(), ego_request[0,0], val_feats_to_send[0,next_id], rospy.get_rostime())
        
        # self.tcp_trans.tensor2Commsg(self.drone_id, self.round_id, shift_mat_tcp, require_maps, features_map_tcp, time_stamp)
        self.tcp_pub.publish(tcp_msg)
    ###################### TO DO #####################
    #       callback function of the TCP-decoder     #
    ##################################################
    def dets_process(self,output,shift_mats):
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg']
        angle = output['angle']
        dets = ctdet_decode(hm, wh, map_scale=1.0, shift_mats=shift_mats, reg=reg, angle=angle, cat_spec_wh=False, K=100)
        return dets
    
    def show_results(self,output, shift_mats, scale = 1.0):
        dets = self.dets_process(output, shift_mats)
        results = []
        detections = []
        meta = self.meta[scale]
        if isinstance(meta, list):
            updated_meta = []
            for cur_meta in meta:
                updated_meta.append({k: v.numpy()[0] for k, v in cur_meta.items()})
            meta = updated_meta
        else:
            meta = {k: v.numpy()[0] for k, v in meta.items()}
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
        detections =  self.Visualization_results(results)
        self.img_transition(self.origin_trans_mat, shift_mats, self.origin_image, detections)

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.num_classes)
        for j in range(1, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32)
            dets[0][j] = dets[0][j].reshape(-1, dets[0][j].shape[-1])
            dets[0][j][:, :(dets[0][j].shape[-1]-1)] /= scale
        return dets[0]
    
    def post_fusion_result(self, trans_mat_input, shift_mats_input, image, origin_polygon_list, co_polygon_list):
        trans_mat = trans_mat_input.copy()
        shift_mat = shift_mats_input.copy()
        print('origin trans_mat: ', trans_mat)
        print('origin shift_mat: ', shift_mat)
        # translation = np.zeros(3,1)
        image_g = self.CoordTrans(image.copy(), trans_mat.copy(), shift_mat.copy())
        image_g = self.vis_cam(image_g.copy(), origin_polygon_list, color=(0, 0, 255), vis_thre=self.vis_score_thre)
        image_g = self.vis_cam(image_g.copy(), co_polygon_list, color=(0, 255, 0), vis_thre=self.vis_score_thre)
        cv2.imshow('BEV_{}'.format(self.drone_id), image_g)
        cv2.waitKey(3)
    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            if len(self.test_scales) > 1:
                if results[j].shape[-1] > 6:
                    polygon_nms(results[j], 0.5)
                else:
                    soft_nms(results[j], Nt=0.5, method=2)
        scores = np.hstack(
            [results[j][:, -1] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, -1] >= thresh)
                results[j] = results[j][keep_inds]
        return results
    
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
                detections.append(detection)
        return detections
    
    def img_transition(self, trans_mat_input, shift_mats_input, image, detections):
        # image_u = cv2.resize(image, (720, 480))
        trans_mat = trans_mat_input.copy()
        shift_mat = shift_mats_input.copy()
        print('origin trans_mat: ', trans_mat)
        print('origin shift_mat: ', shift_mat)
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
    def vis_cam(self, image, polygon_list, color=(127, 255, 0), vis_thre=-1):
    # image = np.ones_like(image) * 255
    # image = np.array(image * 0.85, dtype=np.int32)
    # alpha = np.ones([image.shape[0], image.shape[1], 1]) * 100
    # image = np.concatenate([image, alpha], axis=-1)
    # color = (255, 255, 255)
        for anno in polygon_list:
            print('result anno: ',anno.shape)
            image = cv2.polylines(image, pts=np.int32([anno.reshape(-1, 1, 2)]), isClosed=True, color=color, thickness=2)
        return image  
               
if __name__ == '__main__':
    agent_comm = features_fusion()