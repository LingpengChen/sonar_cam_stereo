# sonar_cam_stereo

/rexrov/rexrov/cameraright/depth/points

roslaunch sonar_cam_stereo world_setup.launch 
roslaunch sonar_cam_stereo launch_rexrov.launch 
rviz -d config/config.rviz 


Camera info
  <xacro:camera namespace="${namespace}" parent_link="${namespace}/base_link" suffix="_ref" update_rate="10" hfov="1.571" width="720" height="480">
    <origin xyz="1.2 0 -0.65" rpy="0 0.262 0"/>
  </xacro:camera>
  <xacro:camera namespace="${namespace}" parent_link="${namespace}/base_link" suffix="_source" update_rate="10" hfov="1.571" width="720" height="480">
    <origin xyz="1.2 -0.2 -0.65" rpy="0 0.262 0"/>
  </xacro:camera>
  

---
header: 
  seq: 7
  stamp: 
    secs: 55
    nsecs: 108000000
  frame_id: "rexrov/cameraright_link_optical"
height: 480
width: 720
distortion_model: "plumb_bob"
D: [0.0, 0.0, 0.0, 0.0, 0.0]
K: [359.92668511204846, 0.0, 360.5, 0.0, 359.92668511204846, 240.5, 0.0, 0.0, 1.0]
R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P: [359.92668511204846, 0.0, 360.5, -0.0, 0.0, 359.92668511204846, 240.5, 0.0, 0.0, 0.0, 1.0, 0.0]
binning_x: 0
binning_y: 0
roi: 
  x_offset: 0
  y_offset: 0
  height: 0
  width: 0
  do_rectify: False
---


header: 
  seq: 6
  stamp: 
    secs: 89
    nsecs: 507000000
  frame_id: "rexrov/cameraright_link_optical"
height: 480
width: 720
distortion_model: "plumb_bob"
D: [0.0, 0.0, 0.0, 0.0, 0.0]
K: [359.92668511204846, 0.0, 360.5, 0.0, 359.92668511204846, 240.5, 0.0, 0.0, 1.0]
R: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
P: [359.92668511204846, 0.0, 360.5, -0.0, 0.0, 359.92668511204846, 240.5, 0.0, 0.0, 0.0, 1.0, 0.0]
binning_x: 0
binning_y: 0
roi: 
  x_offset: 0
  y_offset: 0
  height: 0
  width: 0
  do_rectify: False
---


(528, 512, 3)
基本配置
设备ID: 23415（源设备）
主模式: 2（可能对应于某种预设扫描模式）
频率: 约2.1MHz (2,098,880Hz)，这是一个相对较高的频率，适合近距离高分辨率成像
扫描范围: 1.0米（相对较短的探测距离，适合精细观察）
波束与分辨率特性
波束数量: 256个（这是一个中高端的前视声纳配置）
距离采样点数: 352个
距离分辨率: 约2.8mm (0.002835m)，非常精细的分辨率
总覆盖角度: 根据Oculus典型配置，可能在60°-130°之间
数据特征
数据大小: 图像数据91,520字节，总消息93,568字节
声速: 1487.03 m/s（这是合理的海水声速）
环境参数
温度: 21.55°C
压力: 0.0397 bar（非常浅，基本在水面附近）
盐度: 0.0（设置为淡水环境）
朝向: 1.625°（声纳的方向）
性能设置
Ping速率: 165Hz（这是相对较高的更新率）
增益百分比: 100%（使用最大增益）
Gamma校正: 127（中等设置）

60/12 and 130/20
垂直方向声纳的宽度：np.rad2deg( ping.ping_info.tx_beamwidths[0] )=12.000000333930423
水平方向：[-0.5235987833701112 (-30degree), -0.519060909388374, -0.5145230301094494 ... ] delta_theta=0.26 degree
len=256


range: 3.0
speed_of_sound: 1487.6057254669067
heading: 149.4375
pitch: 14.6875
roll: -2.625
ping_start_time: 2344.329303
data_size: 0
range_resolution: 0.005672736499780472
n_ranges: 528
n_beams: 512
image_offset: 2048
image_size: 272448
message_size: 274496


60 degree
