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