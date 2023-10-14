#include <arpa/inet.h>
#include <netinet/in.h>
#include <ros/init.h>
#include <ros/message_traits.h>
#include <ros/publisher.h>
#include <ros/rate.h>
#include <ros/ros.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <tcp_bridge/ComMessage.h>
#include <tcp_bridge/Mat2d_33.h>
#include <tcp_bridge/Mat2d_conf.h>
#include <tcp_bridge/Mat3d.h>
#include <unistd.h>

#include <boost/thread.hpp>
#include <cstddef>
#include <cstdint>
#include <iostream>
// #define PORT 8080
// #define NEXT_PORT 8083
#define UDP_PORT 8081
#define BUF_LEN 1048576     // 1MB
#define BUF_LEN_SHORT 1024
#define MSG_LEN 296068  // 1KB
using namespace std;

int send_sock_, server_fd_, recv_sock_;
ros::Subscriber swarm_commute_sub_;
ros::Publisher swarm_commute_pub_;
string tcp_ip_;
int drone_id_;
double odom_broadcast_freq_;
char send_buf_[BUF_LEN], recv_buf_[BUF_LEN];
struct sockaddr_in addr_udp_send_;
tcp_bridge::ComMessage::Ptr compressed_msgs_;

int connect_to_next_drone(const char* ip, const int port) {
  /* Connect */
  int sock = 0;
  struct sockaddr_in serv_addr;
  if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    printf("\n Socket creation error \n");
    return -1;
  }

  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(port);

  // Convert IPv4 and IPv6 addresses from text to binary form
  if (inet_pton(AF_INET, ip, &serv_addr.sin_addr) <= 0) {
    printf("\nInvalid address/ Address not supported \n");
    return -1;
  }

  if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
    printf("Tcp connection to drone_%d Failed", drone_id_ + 1);
    return -1;
  }

  char str[INET_ADDRSTRLEN];
  printf("Connect to %s success!",
         inet_ntop(AF_INET, &serv_addr.sin_addr, str, sizeof(str)));

  return sock;
}

int wait_connection_from_previous_drone(const int port, int& server_fd,
                                        int& new_socket) {
  struct sockaddr_in address;
  int opt = 1;
  int addrlen = sizeof(address);

  // Creating socket file descriptor
  if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
    perror("socket failed");
    exit(EXIT_FAILURE);
  }

  // Forcefully attaching socket to the port
  if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
                 sizeof(opt))) {
    perror("setsockopt");
    exit(EXIT_FAILURE);
  }
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(port);

  // Forcefully attaching socket to the port
  if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
    perror("bind failed");
    exit(EXIT_FAILURE);
  }
  if (listen(server_fd, 3) < 0) {
    perror("listen");
    exit(EXIT_FAILURE);
  }
  if ((new_socket = accept(server_fd, (struct sockaddr*)&address,
                           (socklen_t*)&addrlen)) < 0) {
    perror("accept");
    exit(EXIT_FAILURE);
  }

  char str[INET_ADDRSTRLEN];
  printf("Receive tcp connection from %s",
         inet_ntop(AF_INET, &address.sin_addr, str, sizeof(str)));

  return new_socket;
}

int serializeMessage(const tcp_bridge::ComMessage::Ptr& msg) {
  char* ptr = send_buf_;
  ptr+= sizeof(int);
  // tcp_bridge::ComMessage::Ptr msg = compress(ori_msg);
  unsigned long total_len = 0;

  total_len += 3 * sizeof(int32_t);
  total_len += (int)msg->header.frame_id.length() * sizeof(char);
  total_len += sizeof(uint32_t);
  total_len += sizeof(double);

  total_len += sizeof(int32_t);
  for (int i = 0; i < 9; ++i) {
    total_len += sizeof(int32_t);
    for (size_t j = 0; j < msg->mat2d_33[i].num; ++j) {
      total_len += sizeof(int32_t) + sizeof(double);
    }
  }

  total_len += 3 * sizeof(int32_t);
  for (size_t i = 0; i < msg->mat2d_conf.num; ++i) {
    total_len += 2 * sizeof(int32_t) + sizeof(double);
  }

  total_len += sizeof(int32_t);
  for (size_t i = 0; i < msg->mat3d.num; ++i) {
    total_len += 3 * sizeof(int32_t) + sizeof(double);
  }

  if (total_len + 1 > BUF_LEN) {
    printf("[bridge_node] Topic is too large, please enlarge BUF_LEN");
    return -1;
  }

  *((int32_t*)ptr) = msg->drone_id;
  ptr += sizeof(int32_t);
  *((int32_t*)ptr) = msg->turns;
  ptr += sizeof(int32_t);

  int32_t len = (int)msg->header.frame_id.length();
  *((int32_t*)ptr) = len;
  ptr += sizeof(int32_t);
  for (int i = 0; i < len; i++) {
    *((char*)ptr) = msg->header.frame_id[i];
    ptr += sizeof(char);
  }
  *((uint32_t*)ptr) = msg->header.seq;
  ptr += sizeof(uint32_t);
  *((double*)ptr) = msg->header.stamp.toSec();
  ptr += sizeof(double);

  *((int32_t*)ptr) = msg->mat2d_num;
  ptr += sizeof(int32_t);
  for (int i = 0; i < 9; ++i) {
    *((int32_t*)ptr) = msg->mat2d_33[i].num;
    ptr += sizeof(int32_t);
    for (int j = 0; j < msg->mat2d_33[i].num; ++j) {
      *((int32_t*)ptr) = msg->mat2d_33[i].index[j];
      ptr += sizeof(int32_t);
      *((double*)ptr) = msg->mat2d_33[i].val[j];
      ptr += sizeof(double);
    }
  }
  *((int32_t*)ptr) = msg->mat2d_conf.size[0];
  ptr += sizeof(int32_t);
  *((int32_t*)ptr) = msg->mat2d_conf.size[1];
  ptr += sizeof(int32_t);
  *((int32_t*)ptr) = msg->mat2d_conf.num;
  ptr += sizeof(int32_t);
  for (int i = 0; i < msg->mat2d_conf.num; ++i) {
    *((int32_t*)ptr) = msg->mat2d_conf.index[2 * i];
    ptr += sizeof(int32_t);
    *((int32_t*)ptr) = msg->mat2d_conf.index[2 * i + 1];
    ptr += sizeof(int32_t);
    *((double*)ptr) = msg->mat2d_conf.val[i];
    ptr += sizeof(double);
  }

  *((int32_t*)ptr) = msg->mat3d.num;
  ptr += sizeof(int32_t);
  *((int32_t*)ptr) = msg->mat3d.size[0];
  ptr += sizeof(int32_t);
  *((int32_t*)ptr) = msg->mat3d.size[1];
  ptr += sizeof(int32_t);
  *((int32_t*)ptr) = msg->mat3d.size[2];
  ptr += sizeof(int32_t);
  for (int i = 0; i < msg->mat3d.num; ++i) {
    *((int32_t*)ptr) = msg->mat3d.index[3 * i];
    ptr += sizeof(int32_t);
    *((int32_t*)ptr) = msg->mat3d.index[3 * i + 1];
    ptr += sizeof(int32_t);
    *((int32_t*)ptr) = msg->mat3d.index[3 * i + 2];
    ptr += sizeof(int32_t);
    *((double*)ptr) = msg->mat3d.val[i];
    ptr += sizeof(double);
  }

  return ptr - send_buf_;
}

int deserializeMessage(const tcp_bridge::ComMessage::Ptr& msg) {
  char* ptr = recv_buf_;
  msg->drone_id = *((int32_t*)ptr);
  ptr += sizeof(int32_t);
  msg->turns = *((int32_t*)ptr);
  ptr += sizeof(int32_t);

  int32_t len = *((int32_t*)ptr);
  ptr += sizeof(int32_t);
  for (int i = 0; i < len; ++i) {
    msg->header.frame_id.push_back(*(const char*)ptr);
    ptr += sizeof(char);
  }
  msg->header.seq = *((uint32_t*)ptr);
  ptr += sizeof(uint32_t);
  msg->header.stamp.fromSec(*((double*)ptr));
  ptr += sizeof(double);

  msg->mat2d_num = *((int32_t*)ptr);
  ptr += sizeof(int32_t);
  for (int i = 0; i < 9; i++) {
    msg->mat2d_33[i].num = *((int32_t*)ptr);
    ptr += sizeof(int32_t);
    for (int j = 0; j < msg->mat2d_33[i].num; ++j) {
      msg->mat2d_33[i].index.push_back(*((int32_t*)ptr));
      ptr += sizeof(int32_t);
      msg->mat2d_33[i].val.push_back(*((double*)ptr));
      ptr += sizeof(double);
    }
  }
  msg->mat2d_conf.size[0] = *((int32_t*)ptr);
  ptr += sizeof(int32_t);
  msg->mat2d_conf.size[1] = *((int32_t*)ptr);
  ptr += sizeof(int32_t);
  msg->mat2d_conf.num = *((int32_t*)ptr);
  ptr += sizeof(int32_t);
  for (int i = 0; i < msg->mat2d_conf.num; ++i) {
    msg->mat2d_conf.index.push_back(*((int32_t*)ptr));
    ptr += sizeof(int32_t);
    msg->mat2d_conf.index.push_back(*((int32_t*)ptr));
    ptr += sizeof(int32_t);
    msg->mat2d_conf.val.push_back(*((double*)ptr));
    ptr += sizeof(double);
  }

  msg->mat3d.num = *((int32_t*)ptr);
  ptr += sizeof(int32_t);
  msg->mat3d.size[0] = *((int32_t*)ptr);
  ptr += sizeof(int32_t);
  msg->mat3d.size[1] = *((int32_t*)ptr);
  ptr += sizeof(int32_t);
  msg->mat3d.size[2] = *((int32_t*)ptr);
  ptr += sizeof(int32_t);
  for (int i = 0; i < msg->mat3d.num; ++i) {
    msg->mat3d.index.push_back(*((int32_t*)ptr));
    ptr += sizeof(int32_t);
    msg->mat3d.index.push_back(*((int32_t*)ptr));
    ptr += sizeof(int32_t);
    msg->mat3d.index.push_back(*((int32_t*)ptr));
    ptr += sizeof(int32_t);
    msg->mat3d.val.push_back(*((double*)ptr));
    ptr += sizeof(double);
  }
  return ptr - recv_buf_;
}

void multitraj_sub_tcp_cb(const tcp_bridge::ComMessage::Ptr& msg) {
  int len = serializeMessage(msg);
  memcpy(send_buf_, &len , sizeof(int));

  if (send(send_sock_, send_buf_, len , 0) <= 0) {
    printf("TCP SEND ERROR!!!");
  }
}

void server_fun(int PORT) {
  int valread;

  // Connect
    if (wait_connection_from_previous_drone(PORT, server_fd_, recv_sock_) < 0) {
    printf("[bridge_node]Socket recever creation error!");
    exit(EXIT_FAILURE);
  }
  int len = 0, size_len = 0, decode_len= 0;
  while (true) {
    size_len = recv(recv_sock_, &len, sizeof(int), MSG_WAITALL);
    valread = recv(recv_sock_, recv_buf_, len - sizeof(int), MSG_WAITALL);
    if (valread <= 0) {
      printf("Received message length <= 0, maybe connection has lost");
      close(recv_sock_);
      close(server_fd_);
      return;
    }
    decode_len = deserializeMessage(compressed_msgs_);
    if (valread == decode_len) {
      swarm_commute_pub_.publish(*compressed_msgs_);
    } else {
      continue;
    }
  }
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "tcp_bridge");
  ros::NodeHandle nh("~");
  std::string ns = ros::this_node::getNamespace();
  nh.param("next_drone_ip", tcp_ip_, string("127.0.0.1"));
  nh.param("drone_id", drone_id_, -1);
  int PORT, NEXT_PORT;
  nh.param("PORT", PORT, 8080); 
  nh.param("NEXT_PORT", NEXT_PORT, 8081);
  nh.getParam(ns + "/ego_PORT", PORT);
  nh.getParam(ns + "/other_PORT", NEXT_PORT);
  nh.getParam(ns + "/Drone_id", drone_id_);
  compressed_msgs_.reset(new tcp_bridge::ComMessage);
  int next_drone_id_ = 0; 
  if(drone_id_ == 0) {
    next_drone_id_ = 1;
  }
  if (drone_id_ == -1) {
    printf("Wrong drone_id!");
    exit(EXIT_FAILURE);
  }

  string sub_commute_topic_name = 
     string("/drone_") + std::to_string(drone_id_) + string("_to_drone_")+ std::to_string(next_drone_id_) + string("_sending");
  swarm_commute_sub_ =
      nh.subscribe(sub_commute_topic_name.c_str(), 10, multitraj_sub_tcp_cb);

  string pub_commute_topic_name =
      string("/drone_") + std::to_string(drone_id_) + string("_recive");
  swarm_commute_pub_ =
      nh.advertise<tcp_bridge::ComMessage>(pub_commute_topic_name.c_str(), 10);
  //Connection
  boost::thread recv_thd(server_fun,PORT);
  recv_thd.detach();
  ros::Duration(0.1).sleep();

  // TCP connect
  send_sock_ = connect_to_next_drone(tcp_ip_.c_str(), NEXT_PORT);

  cout << "[tcp_bridge] start running" << endl;

  // Test
   //ros::Rate rate(10);
   //while (ros::ok()) {
     // std::cout << "start======" << std::endl;
     //tcp_bridge::ComMessage test_msg;
     //test_msg.drone_id = 6;
     //test_msg.header.seq = 8;
     //test_msg.header.stamp = ros::Time::now();
     //test_msg.header.frame_id = "earth";
     //test_msg.mat2d_num = 1;
     //for (int i = 0; i < 1; i++) {
     //  test_msg.mat2d_33[i].num = 0;
     //}
     //test_msg.mat2d_conf.size = {1, 1};
     //test_msg.mat2d_conf.num = 0;
     //test_msg.mat3d.size = {1, 1, 1};
     //test_msg.mat3d.num = 0;

     //swarm_commute_pub_.publish(test_msg);
     //ros::spinOnce();
     //rate.sleep();
   //}
  ros::spin();

  close(send_sock_);
  close(recv_sock_);
  close(server_fd_);

  return 0;
}
