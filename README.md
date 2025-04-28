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
  