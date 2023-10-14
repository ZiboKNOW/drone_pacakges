 #include "Uwb_Location/trilateration.h"
#include "Uwb_Location/uwb.h"
#include "chrono"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "std_msgs/String.h" //ros定义的String数据类型
#include <iostream>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <serial/serial.h>
#include <string.h>
#include <string>

using namespace std;
using namespace cv;
unsigned char receive_buf[200] = {0};
vec3d report;
Quaternion q;
int result = 0; 
float velocityac[3],angleac[3];
Quaternion Q;
bool init_flag = false;

#define MAX_DATA_NUM	1024	//传消息内容最大长度
#define DataHead        'm'       
#define DataTail        '\n'   
#define baseHeight 1.23
unsigned char BufDataFromCtrl[MAX_DATA_NUM];
int BufCtrlPosit_w = 0, BufCtrlPosit_r = 0;
int DataRecord=0, rcvsign = 0;
std::string img_name = "final.png";
vec3d anchorArray[8];

void arrow(cv::Mat &img, cv::Point p1, cv::Point p2, cv::Scalar &color,
           int thickness, int alpha) {
  Point arrow;
  const double PI = 3.1415926;
  float len = sqrt((float)((p2.y - p1.y) * (p2.y - p1.y) +
                           (p1.x - p2.x) * (p1.x - p2.x))); // 计算俩点间距离
  double angle =
      atan2((double)(p2.y - p1.y),
            (double)(p2.x - p1.x)); //**注意这里角度的计算结果为弧度制**
  line(img, p1, p2, color, thickness);

  /*下边画箭头的俩根斜线*/
  arrow.x = (int)(p2.x - len * cos(angle + PI * alpha / 180));
  arrow.y = (int)(p2.y - len * sin(angle + PI * alpha / 180));
  line(img, arrow, p2, color, thickness);
  arrow.x = (int)(p2.x - len * cos(angle - PI * alpha / 180));
  arrow.y = (int)(p2.y - len * sin(angle - PI * alpha / 180));
  line(img, arrow, p2, color, thickness);
}

// Initialize a canvas(for painting)
void init_draw(const double x, const double y) {
  Mat CSYS = Mat::zeros(Size(1000, 1500), CV_8UC3);
  CSYS.setTo(255);
  Scalar color_black(0, 0, 0), color_word(0, 0, 0), color_red(47, 0, 255),
      color_green(47, 225, 0), color_blue(225, 47, 0),
      color_white(255, 255, 255);                                // 设置颜色
  line(CSYS, Point(100, 100), Point(100, 1420), color_black, 3); // Y轴
  line(CSYS, Point(100, 100), Point(900, 100), color_black, 3);  // X轴
  arrow(CSYS, Point(900, 100), Point(916, 100), color_black, 3, 30); // X轴箭头
  arrow(CSYS, Point(100, 1400), Point(100, 1416), color_black, 3,
        30); // Y轴箭头
  putText(CSYS, "x", Point(910, 80), FONT_HERSHEY_SIMPLEX, 1.5, color_black,
          3); // 坐标轴文本
  putText(CSYS, "0", Point(65, 80), FONT_HERSHEY_SIMPLEX, 1.5, color_black, 3);
  putText(CSYS, "y", Point(50, 1430), FONT_HERSHEY_SIMPLEX, 1.5, color_black,
          3);
  line(CSYS, Point(100, 228), Point(115, 228), color_black, 3);
  putText(CSYS, "4", Point(60, 236), FONT_HERSHEY_SIMPLEX, 0.7, color_black, 3);
  line(CSYS, Point(100, 356), Point(115, 356), color_black, 3);
  putText(CSYS, "8", Point(60, 364), FONT_HERSHEY_SIMPLEX, 0.7, color_black, 3);
  line(CSYS, Point(100, 484), Point(115, 484), color_black, 3);
  putText(CSYS, "12", Point(60, 492), FONT_HERSHEY_SIMPLEX, 0.7, color_black,
          3);
  line(CSYS, Point(100, 612), Point(115, 612), color_black, 3);
  putText(CSYS, "16", Point(60, 620), FONT_HERSHEY_SIMPLEX, 0.7, color_black,
          3);
  line(CSYS, Point(100, 740), Point(115, 740), color_black, 3);
  putText(CSYS, "20", Point(60, 748), FONT_HERSHEY_SIMPLEX, 0.7, color_black,
          3);
  line(CSYS, Point(100, 868), Point(115, 868), color_black, 3);
  putText(CSYS, "24", Point(60, 876), FONT_HERSHEY_SIMPLEX, 0.7, color_black,
          3);
  line(CSYS, Point(100, 996), Point(115, 996), color_black, 3);
  putText(CSYS, "28", Point(60, 1004), FONT_HERSHEY_SIMPLEX, 0.7, color_black,
          3);
  line(CSYS, Point(100, 1124), Point(115, 1124), color_black, 3);
  putText(CSYS, "32", Point(60, 1132), FONT_HERSHEY_SIMPLEX, 0.7, color_black,
          3);
  line(CSYS, Point(100, 1252), Point(115, 1252), color_black, 3);
  putText(CSYS, "36", Point(60, 1260), FONT_HERSHEY_SIMPLEX, 0.7, color_black,
          3);
  line(CSYS, Point(100, 1380), Point(115, 1380), color_black, 3);
  putText(CSYS, "40", Point(60, 1388), FONT_HERSHEY_SIMPLEX, 0.7, color_black,
          3); // y轴刻度
  line(CSYS, Point(228, 100), Point(228, 115), color_black, 3);
  putText(CSYS, "4", Point(220, 70), FONT_HERSHEY_SIMPLEX, 0.7, color_black, 3);
  line(CSYS, Point(356, 100), Point(356, 115), color_black, 3);
  putText(CSYS, "8", Point(348, 70), FONT_HERSHEY_SIMPLEX, 0.7, color_black, 3);
  line(CSYS, Point(484, 100), Point(484, 115), color_black, 3);
  putText(CSYS, "12", Point(476, 70), FONT_HERSHEY_SIMPLEX, 0.7, color_black,
          3);
  line(CSYS, Point(612, 100), Point(612, 115), color_black, 3);
  putText(CSYS, "16", Point(604, 70), FONT_HERSHEY_SIMPLEX, 0.7, color_black,
          3);
  line(CSYS, Point(740, 100), Point(740, 115), color_black, 3);
  putText(CSYS, "20", Point(732, 70), FONT_HERSHEY_SIMPLEX, 0.7, color_black,
          3); // x轴刻度
  line(CSYS, Point(868, 100), Point(868, 115), color_black, 3);
  putText(CSYS, "24", Point(860, 70), FONT_HERSHEY_SIMPLEX, 0.7, color_black,
          3);

  circle(
      CSYS,
      Point_<float>((anchorArray[0].x + 5) * 32, (anchorArray[0].x + 5) * 32),
      7, color_red, 3); // 画三/四个基站
  circle(
      CSYS,
      Point_<float>((anchorArray[1].x + 5) * 32, (anchorArray[1].y + 5) * 32),
      7, color_red, 3);
  circle(
      CSYS,
      Point_<float>((anchorArray[2].x + 5) * 32, (anchorArray[2].y + 5) * 32),
      7, color_red, 3);
  if (anchorArray[3].x * 100 != 10000.0) {
    circle(
        CSYS,
        Point_<float>((anchorArray[3].x + 5) * 32, (anchorArray[3].y + 5) * 32),
        7, color_red, 3);
  }
  circle(CSYS, Point_<float>((x + 5) * 32, (y + 5) * 32), 5, color_blue,
         -1); // 画定位点

  // Groundtruth
  // line(CSYS,Point(220,500),Point(220,420),color,2);
  // line(CSYS,Point(220,420),Point(380,420),color,2);
  // line(CSYS,Point(380,420),Point(380,500),color,2);
  // putText(CSYS,"t1",
  // Point(370,530),FONT_HERSHEY_PLAIN,1,color_word,1);//函数文本
  // putText(CSYS,"t0",Point(200,530),FONT_HERSHEY_PLAIN,1,color_word,1);

  namedWindow("实时监控");
  vector<int> compression_params;
  compression_params.push_back(IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);
  imwrite(img_name, CSYS, compression_params);
  imshow("实时监控", CSYS);
  // corresponding to the period of ranging
  waitKey(112);
  return;
}

void draw(string img, const double x, const double y) {
  Mat Tmp = imread(img);
  Scalar color_blue(225, 47, 0);
  circle(Tmp, Point_<float>((x + 5) * 32, (y + 5) * 32), 5, color_blue,
         -1); // 画定位点

  // namedWindow("实时监控");
  vector<int> compression_params;
  compression_params.push_back(IMWRITE_PNG_COMPRESSION);
  compression_params.push_back(9);
  imwrite(img_name, Tmp, compression_params);
  imshow("实时监控", Tmp);
  // corresponding to the period of ranging
  waitKey(112);
  return;
}

void receive_deal_func()
{
    
    int range[8] = {-1};
    if((receive_buf[0] == 'm') && (receive_buf[1] == 'c'))
    {
        int aid, tid, lnum, seq, mask;
        int rangetime;
        char role;
        int data_len = strlen((char*)receive_buf);
        //printf("lenmc = %d\n", data_len);
        if(data_len == 106)
        {
            int n = sscanf((char*)receive_buf,"mc %x %x %x %x %x %x %x %x %x %x %x %x %c%d:%d", 
                            &mask, 
                            &range[0], &range[1], &range[2], &range[3], 
                            &range[4], &range[5], &range[6], &range[7],
                            &lnum, &seq, &rangetime, &role, &tid, &aid);
            printf("mask=0x%02x\nrange[0]=%d(mm)\nrange[1]=%d(mm)\nrange[2]=%d(mm)\nrange[3]=%d(mm)\nrange[4]=%d(mm)\nrange[5]=%d(mm)\nrange[6]=%d(mm)\nrange[7]=%d(mm)\r\n",
                                mask, range[0], range[1], range[2], range[3], 
                                      range[4], range[5], range[6], range[7]);
        }
        else if(data_len == 70)
        {
            int n = sscanf((char*)receive_buf,"mc %x %x %x %x %x %x %x %x %c%d:%d", 
                            &mask, 
                            &range[0], &range[1], &range[2], &range[3], 
                            &lnum, &seq, &rangetime, &role, &tid, &aid);
            printf("mask=0x%02x\nrange[0]=%d(mm)\nrange[1]=%d(mm)\nrange[2]=%d(mm)\nrange[3]=%d(mm)\r\n",
                                mask, range[0], range[1], range[2], range[3]);
        }
        else
        {
            return;
        }
    }
    else if((receive_buf[0] == 'm') && (receive_buf[1] == 'i'))
    {
        float rangetime2;
        float acc[3], gyro[3];

        //mi,981.937,0.63,NULL,NULL,NULL,-2.777783,1.655664,9.075048,-0.004788,-0.014364,-0.001596,T0     //13
        //mi,3.710,0.55,NULL,NULL,NULL,NULL,NULL,NULL,NULL,-1.327881,0.653174,9.577490,-0.004788,-0.013300,-0.002128,T0    //17
        char *ptr, *retptr;
        ptr = (char*)receive_buf;
        char cut_data[30][12];
        int cut_count = 0;

        while((retptr = strtok(ptr,",")) != NULL )
        {
            //printf("%s\n", retptr);
            strcpy(cut_data[cut_count], retptr);
            ptr = NULL;
            cut_count++;
            if(cut_count >= 29)
                break;
        }

        rangetime2 = atof(cut_data[1]);
        
        if(cut_count == 13)  //4anchors
        {
            for(int i = 0; i < 4; i++)
            {
                if(strcmp(cut_data[i+2], "NULL"))
                {
                    range[i] = atof(cut_data[i+2]) * 1000;
                }
                else
                {
                    range[i] = -1;
                }
            }

            for(int i = 0; i < 3; i++)
            {
                acc[i] = atof(cut_data[i+6]);
            }

            for(int i = 0; i < 3; i++)
            {
                gyro[i] = atof(cut_data[i+9]);
            }

            printf("rangetime = %.3f\n", rangetime2);
            printf("range[0] = %d\n", range[0]);
            printf("range[1] = %d\n", range[1]);
            printf("range[2] = %d\n", range[2]);
            printf("range[3] = %d\n", range[3]);
            printf("acc[0] = %.3f\n", acc[0]);
            printf("acc[1] = %.3f\n", acc[1]);
            printf("acc[2] = %.3f\n", acc[2]);
            printf("gyro[0] = %.3f\n", gyro[0]);
            printf("gyro[1] = %.3f\n", gyro[1]);
            printf("gyro[2] = %.3f\n", gyro[2]);
        }
        else if(cut_count == 17)  //8anchors
        {
            for(int i = 0; i < 8; i++)
            {
                if(strcmp(cut_data[i+2], "NULL"))
                {
                    range[i] = atof(cut_data[i+2]) * 1000;
                }
                else
                {
                    range[i] = -1;
                }
            }

            for(int i = 0; i < 3; i++)
            {
                acc[i] = atof(cut_data[i+6+4]);
            }

            for(int i = 0; i < 3; i++)
            {
                gyro[i] = atof(cut_data[i+9+4]);
            }

            printf("rangetime = %.3f\n", rangetime2);
            printf("range[0] = %d\n", range[0]);
            printf("range[1] = %d\n", range[1]);
            printf("range[2] = %d\n", range[2]);
            printf("range[3] = %d\n", range[3]);
            printf("range[4] = %d\n", range[4]);
            printf("range[5] = %d\n", range[5]);
            printf("range[6] = %d\n", range[6]);
            printf("range[7] = %d\n", range[7]);
            printf("acc[0] = %.3f\n", acc[0]);
            printf("acc[1] = %.3f\n", acc[1]);
            printf("acc[2] = %.3f\n", acc[2]);
            printf("gyro[0] = %.3f\n", gyro[0]);
            printf("gyro[1] = %.3f\n", gyro[1]);
            printf("gyro[2] = %.3f\n", gyro[2]);
        }
        else
        {
            return;
        }
    }
    else
    {
        puts("no range message");
        return;
    }

   

    result = GetLocation(&report, &anchorArray[0], &range[0]);

    printf("result = %d\n",result);
    if(result != 3 && result != 4){
        printf("anchorArray[0].x = %f\nanchorArray[1].x = %f\nanchorArray[2].x = %f\nanchorArray[3].x = %f\n",anchorArray[0].x,anchorArray[1].x,anchorArray[2].x,anchorArray[3].x);
    }

    printf("x = %f\n",report.x);
    printf("y = %f\n",report.y);
    printf("z = %f\n",report.z);

}


void CtrlSerDataDeal()
{
    unsigned char middata = 0;
    static unsigned char dataTmp[MAX_DATA_NUM] = {0};

    while(BufCtrlPosit_r != BufCtrlPosit_w)
    {
        middata = BufDataFromCtrl[BufCtrlPosit_r];
        BufCtrlPosit_r = (BufCtrlPosit_r==MAX_DATA_NUM-1)? 0 : (BufCtrlPosit_r+1);

        if(((middata == DataHead))&&(rcvsign == 0))//收到头
        {
            rcvsign = 1;//开始了一个数据帧
            dataTmp[DataRecord++] = middata;//数据帧接收中
        }
        else if((middata != DataTail)&&(rcvsign == 1))
        {
            dataTmp[DataRecord++] = middata;//数据帧接收中
        }
        else if((middata == DataTail)&&(rcvsign == 1))//收到尾
        {
            if(DataRecord != 1)
            {
                rcvsign = 0;
                dataTmp[DataRecord++] = middata;
                dataTmp[DataRecord] = '\0';

                strncpy((char*)receive_buf, (char*)dataTmp, DataRecord);
                printf("receive_buf = %slen = %d\n", receive_buf, DataRecord);
                receive_deal_func(); /*调用处理函数*/
                bzero(receive_buf, sizeof(receive_buf));

                DataRecord = 0;
            }
        }
    }
}


int main(int argc, char** argv)
{
    setlocale(LC_ALL,"");
	std_msgs::String msg;
	std_msgs::String  msg_mc;
	int  data_size;
	int n;
	int cnt = 0;
    cv::Mat image;
    ros::init(argc, argv, "uwb_imu_node");//发布imu,uwb节点
    //创建句柄（虽然后面没用到这个句柄，但如果不创建，运行时进程会出错）
    ros::NodeHandle nh;
    ros::NodeHandle nh1;
    ros::Publisher uwb_publisher = nh.advertise<Uwb_Location::uwb>("/uwb/data", 1000);//发布uwb数据  话题名 队列大小
    ros::Publisher IMU_read_pub = nh.advertise<sensor_msgs::Imu>("imu/data", 1000);//发布imu话题

    //创建一个serial类
    serial::Serial sp;
    //创建timeout
    serial::Timeout to = serial::Timeout::simpleTimeout(11);
    //设置要打开的串口名称
    sp.setPort("/dev/ttyUSB0");
    //设置串口通信的波特率
    sp.setBaudrate(115200);
    //串口设置timeout
    sp.setTimeout(to);

 
    try
    {
        //打开串口
        sp.open();
    }
    catch(serial::IOException& e)
    {
        ROS_ERROR_STREAM("Unable to open port.");
        return -1;
    }
    
    //判断串口是否打开成功
    if(sp.isOpen())
    {
        ROS_INFO_STREAM("/dev/ttyUSB0 is opened.");
    }
    else
    {
        return -1;
    }
    
    //ros::Rate loop_rate(11);

    //发布uwb话题
    Uwb_Location::uwb uwb_data;
    //打包IMU数据
    sensor_msgs::Imu imu_data;

    // 视频制作
    VideoWriter writer;
    int coder = VideoWriter::fourcc('M', 'J', 'P', 'G'); // 选择编码格式

    double fps = 8.9;                // 设置视频帧率
    string filename = "trailor.avi"; // 保存的视频文件名称
    writer.open(filename, coder, fps, Size(1000, 1500),
                true); // 创建保存视频文件的视频流
     // A0 uint:m
    anchorArray[0].x = 0.0;
    anchorArray[0].y = 0.0;
    anchorArray[0].z = baseHeight;
    // A1 uint:m
    anchorArray[1].x = 6.6;
    anchorArray[1].y = 0.0;
    anchorArray[1].z = baseHeight;
    // A2 uint:m
    anchorArray[2].x = 0.0;
    anchorArray[2].y = 21.4;
    anchorArray[2].z = baseHeight;
    // A3 uint:m
    anchorArray[3].x = 6.6;
    anchorArray[3].y = 21.4;
    anchorArray[3].z = 2.76;

    while(ros::ok())
    {
        //获取缓冲区内的字节数
        size_t len = sp.available();
        if(len > 0)
        {
            unsigned char usart_buf[1024]={0};
            sp.read(usart_buf, len);
            //printf("uart_data = %s\n", usart_buf);

            unsigned char *pbuf;
            unsigned char buf[2014] = {0};

            pbuf = (unsigned char *)usart_buf;
            memcpy(&buf[0], pbuf, len);

            int reallength = len;
            int i;
            if(reallength != 0)
            {
                for( i=0; i < reallength; i++)
                {
                    BufDataFromCtrl[BufCtrlPosit_w] = buf[i];

                    BufCtrlPosit_w = (BufCtrlPosit_w==(MAX_DATA_NUM-1))? 0 : (1 + BufCtrlPosit_w);
                }
            }
            CtrlSerDataDeal();
            if (!init_flag) {
                init_draw(report.x, report.y);
                init_flag = true;
            }
            // 获取图像
            image = imread(img_name);
            // 检测是否成功获取图像
            if(image.empty()){// 判断有没有读取图像成功
                printf("没有获取到图像\n");
                return -1;
            }

            if(!writer.isOpened()){
                printf("打开视频文件失败，请确认是否为合法输入\n");
                return -1;
            }
            writer.write(image);// 把图像写入视频流
//---------------------------------UWB----------------------------------------------------
            uwb_data.time=ros::Time::now();
            uwb_data.x = report.x;
            uwb_data.y = report.y;
            uwb_data.z = report.z;
            //printf("tag.x=%.3f\r\ntag.y=%.3f\r\ntag.z=%.3f\r\n",uwb_data.x,uwb_data.y,uwb_data.z);
            if(init_flag){
                draw(img_name, report.x, report.y);
            }

//--------------------------------------IMU------------------------------------------------
            //发布imu话题
            imu_data.header.stamp =uwb_data.time;
            imu_data.header.frame_id = "base_link";
            imu_data.linear_acceleration.x=velocityac[0];
            imu_data.linear_acceleration.y=velocityac[1];
            imu_data.linear_acceleration.z=velocityac[2];

            //角速度
            imu_data.angular_velocity.x = angleac[0]; 
            imu_data.angular_velocity.y = angleac[1]; 
            imu_data.angular_velocity.z = angleac[2];

            //四元数
            imu_data.orientation.x = Q.q1;
            imu_data.orientation.y = Q.q2;
            imu_data.orientation.z = Q.q3;
            imu_data.orientation.w = Q.q0;

//--------------------------------------话题发布------------------------------------
            uwb_publisher.publish(uwb_data);
            IMU_read_pub.publish(imu_data);
        }

        ros::spinOnce(); 
        //loop_rate.sleep();
    }
    //关闭串口
    sp.close();
    // 结束视频
    writer.release();
    return 0;
}
