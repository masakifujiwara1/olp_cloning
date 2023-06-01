for i in `seq 1`
do
  roslaunch nav_cloning nav_cloning_sim.launch script:=nav_cloning_with_direction_node.py mode:=selected_training world_name:=tsudanuma_scan.world map_file:=cit_3f_map.yaml waypoints_file:=cit3f_way.yaml dist_err:=0.6 initial_pose_x:=-5.0 initial_pose_y:=7.7 initial_pose_a:=3.14 use_waypoint_nav:=false robot_x:=20.38 robot_y:=0.76 robot_Y:=-1.57
  sleep 10
done
