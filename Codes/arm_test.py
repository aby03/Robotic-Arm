import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
print "============ Starting tutorial setup"
moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('move_group_python_interface_tutorial',anonymous=True)
robot = moveit_commander.RobotCommander()
scene = moveit_commander.PlanningSceneInterface()
group = moveit_commander.MoveGroupCommander("arm_1")
display_trajectory_publisher = rospy.Publisher(
                                    '/move_group/display_planned_path',
                                    moveit_msgs.msg.DisplayTrajectory)
print "============ Waiting for RVIZ..."
print "============ Starting tutorial "
print "============ Reference frame: %s" % group.get_planning_frame()
print "============ Reference frame: %s" % group.get_end_effector_link()
print "============ Robot Groups:"
print robot.get_group_names()
print "============ Printing robot state"
print robot.get_current_state()
print "============"
print "============ Generating plan 1"
pose_target = geometry_msgs.msg.Pose()
#pose_target = group.get_current_pose()
#print pose_target
pose_target.position.x = 0.112887985
pose_target.position.y = 0.2594417410132
pose_target.position.z = 0.435656721551
pose_target.orientation.x = 0.410398597913
pose_target.orientation.y = 0.276952949658
pose_target.orientation.z = -0.851386508607
pose_target.orientation.w = 0.247473140566
group.set_pose_target(pose_target)
group.set_goal_tolerance(0.01);
plan1 = group.plan()
group.go(pose_target)

print "============ Waiting while RVIZ displays plan1..."
