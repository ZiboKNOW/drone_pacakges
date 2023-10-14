import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
import socket
import time
from sensor_msgs.msg import Image
from std_msgs.msg import String
from drone_detection.msg import mount

def HEXtoDEC(dec_data):
    if(dec_data > 1800): 
        dec_data = 2**16-dec_data       
        dec_data = 0-dec_data
    return dec_data

class Video_Capture:

    def __init__(self):
        rospy.init_node('Videocapture', anonymous=True)
        self.udp_socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.dest_ip = '192.168.144.25'
        self.dest_port = 37260
        self.init_mount()
    
    def pub_image(self, event):
        _, frame = self.cap.read()
        frame = cv2.resize(frame, (720, 480))
        ros_image= self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        ros_image.header.stamp = rospy.Time.now()
        self.pubIm.publish(ros_image)
    
    def pub_ori(self, event):
        pos_command = "55 66 01 00 00 00 00 0d e8 05"
        hex_data = pos_command.replace(' ','')
        send_command = bytes.fromhex(hex_data)
        self.udp_socket.sendto(send_command,(self.dest_ip,self.dest_port))
        ori_data, server = self.udp_socket.recvfrom(4096)
        #print(ori_data)
        print( 'ori: ' + ''.join(['%02x ' % b for b in ori_data]))

        mount_msg = mount()
        mount_msg.header.stamp = rospy.Time.now()
        Ori = ''.join(['%02x ' % b for b in ori_data])
        ori = Ori.replace(' ','')
        #print(ori)
        mount_msg.yaw = HEXtoDEC(int(ori[18:20]+ori[16:18],16))
        mount_msg.pitch = HEXtoDEC(int(ori[22:24]+ori[20:22],16))
        mount_msg.roll = HEXtoDEC(int(ori[26:28]+ori[24:26],16))
        #print(HEXtoDEC(int(ori[26:28]+ori[24:26],16)))
        self.pubOri.publish(mount_msg)

    def cap_image(self):
        adress = 'rtsp://'+self.dest_ip+':8554/main.264'
        print('adress: ', adress)
        self.cap = cv2.VideoCapture('rtsp://'+self.dest_ip+':8554/main.264')
        self.pubIm =  rospy.Publisher("/drone_image", Image, queue_size=3) 
        self.bridge = CvBridge()
        rospy.Timer(rospy.Duration(0.03), self.pub_image)
        self.pubOri =  rospy.Publisher("/camera_ori", mount, queue_size=30) 
        rospy.Timer(rospy.Duration(0.03), self.pub_ori)
        rospy.spin()

    # def pub_ori(self, ori_data):
    #     # 55 66 02 0c 00 06 00 0d f7 ff 90 fc 00 00 fd ff 7f ff 00 00 89 4b
    #     mount_msg = mount()
    #     mount_msg.header.stamp = rospy.Time.now()
    #     ori = ori_data.replace(" ","")
    #     mount_msg.yaw = int(hex(ori[16:20]),16)
    #     mount_msg.pitch = int(hex(ori[20:24]),16)
    #     mount_msg.roll = int(hex(ori[24:28]),16)
    #     self.pub.publish(mount_msg)

    # def cap_ori(self):
        # pos_command = "55 66 01 00 00 00 00 0d e8 05"
        # hex_data = pos_command.replace(' ','')
        # send_command = bytes.fromhex(hex_data)
        # self.udp_socket.sendto(send_command,(self.dest_ip,self.dest_port))
        # ori_data, server = self.udp_socket.recvfrom(4096)
        # print( 'ori: ' + ''.join(['%02x ' % b for b in ori_data]))
        # self.pub =  rospy.Publisher("/camera_ori", mount, queue_size=3) 
        # rospy.Timer(rospy.Duration(0.03), self.pub_ori(ori_data))
        # rospy.spin()

    def init_mount(self):
        ori_command = "55 66 01 04 00 00 00 0e 0000 0384 b7 43"
        command_lock = "55 66 01 01 00 00 00 0c 03 57 fe"
        reset_cpmmand = "55 66 01 01 00 00 00 08 01 d1 12"
        # ori_command = "55 66 01 02 00 00 00 07 20 20 75 06"
        # ori_command = "55 66 01 02 00 00 00 07 64 64 3d cf"
        hex_data = ori_command.replace(' ','')
        send_command = bytes.fromhex(hex_data)
        self.udp_socket.sendto(send_command,(self.dest_ip,self.dest_port))
        data, server = self.udp_socket.recvfrom(4096)
        print( ''.join(['%02x ' % b for b in data]))
        time.sleep(1)
        hex_data = reset_cpmmand.replace(' ','')
        send_command = bytes.fromhex(hex_data)
        self.udp_socket.sendto(send_command,(self.dest_ip,self.dest_port))
        data, server = self.udp_socket.recvfrom(4096)
        print( ''.join(['%02x ' % b for b in data]))    
        time.sleep(1)
        hex_data = command_lock.replace(' ','')
        send_command = bytes.fromhex(hex_data)
        self.udp_socket.sendto(send_command,(self.dest_ip,self.dest_port))

    def reset_mount(self):
        reset = '55 66 01 01 00 00 00 08 01 d1 12'
        hex_data = reset.replace(' ','')
        send_command = bytes.fromhex(hex_data)
        self.udp_socket.sendto(send_command,(self.dest_ip,self.dest_port))
# cap = cv2.VideoCapture('rtsp://192.168.144.25:8554/main.264')



if __name__ == '__main__':
    video_capture = Video_Capture()
    try:
        video_capture.cap_image()
        #video_capture.cap_ori()
    except KeyboardInterrupt:
        video_capture.reset_mount()
        video_capture.udp_socket.close()

