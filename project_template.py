#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
	get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
	return get_normals_prox(cloud).cluster
	#search in list of dictionaries
def search_dict(key1,value1,key2,list_dicts):

	selected_dict=[element for element in list_dicts if element[key1]==value1][0]#accessing the 1st element in the list of found dictionaries 
	return selected_dict[key2]

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
	yaml_dict = {}
	yaml_dict["test_scene_num"] = test_scene_num.data
	yaml_dict["arm_name"]  = arm_name.data
	yaml_dict["object_name"] = object_name.data
	yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
	yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
	return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
	data_dict = {"object_list": dict_list}
	with open(yaml_filename, 'w') as outfile:
		yaml.dump(data_dict, outfile, default_flow_style=False)


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

	# TODO: Convert ROS msg to PCL data
	cloud=ros_to_pcl(pcl_msg)
	# TODO: Statistical Outlier Filtering
	outlier_filter = cloud.make_statistical_outlier_filter()
	
	# Set the number of neighboring points to analyze for any given point
	outlier_filter.set_mean_k(3)

	# Set threshold scale factor
	x = 0.00001

	# Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
	outlier_filter.set_std_dev_mul_thresh(x)

	# Finally call the filter function for magic
	cloud = outlier_filter.filter()

	#test cloud for debug
	cloud_test=cloud
	# TODO: Voxel Grid Downsampling
	vox = cloud.make_voxel_grid_filter()

	# Choose a voxel (also known as leaf) size
	LEAF_SIZE = 0.005   

	# Set the voxel (or leaf) size  
	vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

	# Call the filter function to obtain the resultant downsampled point cloud
	cloud_filtered = vox.filter()
	# TODO: PassThrough Filter z axis
	passthrough = cloud_filtered.make_passthrough_filter()

	# Assign axis and range to the passthrough filter object.
	filter_axis = 'z'
	passthrough.set_filter_field_name(filter_axis)
	axis_min =0.61
	axis_max =1.1
	passthrough.set_filter_limits(axis_min, axis_max)

	# Finally use the filter function to obtain the resultant point cloud. 
	cloud_filtered = passthrough.filter()
		# TODO: PassThrough Filter y axis
	passthrough = cloud_filtered.make_passthrough_filter()

	# Assign axis and range to the passthrough filter object.
	filter_axis = 'y'
	passthrough.set_filter_field_name(filter_axis)
	axis_min =-0.45
	axis_max =0.45
	passthrough.set_filter_limits(axis_min, axis_max)

	# Finally use the filter function to obtain the resultant point cloud. 
	cloud_filtered = passthrough.filter()

	# TODO: RANSAC Plane Segmentation
	seg = cloud_filtered.make_segmenter()

	# Set the model you wish to fit 
	seg.set_model_type(pcl.SACMODEL_PLANE)
	seg.set_method_type(pcl.SAC_RANSAC)

	# Max distance for a point to be considered fitting the model
	# Experiment with different values for max_distance 
	# for segmenting the table
	max_distance = 0.006
	seg.set_distance_threshold(max_distance)

	# Call the segment function to obtain set of inlier indices and model coefficients

	# TODO: Extract inliers and outliers
	inliers, coefficients = seg.segment()

	cloud_table = cloud_filtered.extract(inliers, negative=False)

	cloud_objects = cloud_filtered.extract(inliers, negative=True)
	# TODO: Euclidean Clustering
	white_cloud=XYZRGB_to_XYZ(cloud_objects)
	tree = white_cloud.make_kdtree()
	# Create a cluster extraction object
	ec = white_cloud.make_EuclideanClusterExtraction()
	# Set tolerances for distance threshold
	# as well as minimum and maximum cluster size (in points)
	# NOTE: These are poor choices of clustering parameters
	# Your task is to experiment and find values that work for segmenting objects.
	ec.set_ClusterTolerance(0.03)#2cm
	ec.set_MinClusterSize(10)
	ec.set_MaxClusterSize(9000)
	# Search the k-d tree for clusters
	ec.set_SearchMethod(tree)
	# Extract indices for each of the discovered clusters
	cluster_indices = ec.Extract()

	# TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
	cluster_color = get_color_list(len(cluster_indices))

	color_cluster_point_list = []

	detected_objects_labels = []
	detected_objects = []
	#Create new cloud containing all clusters, each with unique color
	cluster_cloud = pcl.PointCloud_PointXYZRGB()
	cluster_cloud.from_list(color_cluster_point_list)

	for j, indices in enumerate(cluster_indices):
		# Grab the points for the cluster from the extracted outliers (cloud_objects)
		pcl_cluster = cloud_objects.extract(indices)
		# TODO: convert the cluster from pcl to ROS using helper function
		ros_cluster=pcl_to_ros(pcl_cluster)
		# Extract histogram features
		# TODO: complete this step just as is covered in capture_features.py
		chists = compute_color_histograms(ros_cluster, using_hsv=False)
		normals = get_normals(ros_cluster)
		nhists = compute_normal_histograms(normals)
		feature = np.concatenate((chists, nhists))
		# Make the prediction, retrieve the label for the result
		# and add it to detected_objects_labels list
		prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
		label = encoder.inverse_transform(prediction)[0]
		detected_objects_labels.append(label)

		# Publish a label into RViz
		label_pos = list(white_cloud[indices[0]])
		label_pos[2] += .4
		object_markers_pub.publish(make_label(label,label_pos, j))

		# Add the detected object to the list of detected objects.
		do = DetectedObject()
		do.label = label
		do.cloud = ros_cluster
		detected_objects.append(do)
		for i, indice in enumerate(indices):
			color_cluster_point_list.append([white_cloud[indice][0],
	                                white_cloud[indice][1],
	                                white_cloud[indice][2],
	                                 rgb_to_float(cluster_color[j])])


	# TODO: Convert PCL data to ROS messages
	ros_cluster_test=pcl_to_ros(cloud_test)
	ros_cluster_cloud = pcl_to_ros(cluster_cloud)
	ros_cloud_objects =pcl_to_ros(cloud_objects)  
	ros_cloud_table =pcl_to_ros(cloud_table)
	# TODO: Publish ROS messages
	pcl_cluster_pub.publish(ros_cluster_test)
	pcl_cluster_pub.publish(ros_cluster_cloud)
	pcl_objects_pub.publish(ros_cloud_objects)
	pcl_table_pub.publish(ros_cloud_table)
	rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
	# Publish the list of detected objects
	detected_objects_pub.publish(detected_objects)




        

    

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
	try:
		print ("try pr2_mover")
		pr2_mover(detected_objects)
	except rospy.ROSInterruptException:
		pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # TODO: Initialize variables
    test_scene_num = Int32()#put in the message
    object_name = String()#put in the message
    object_group = String()#put in the message
    arm_name=String()#put in the message

    pick_pose = Pose()#put in the message
    place_pose = Pose()#put in the message
    #modify for different worlds 1,2,3
    test_scene_num.data=1
    #initialise yaml dictionary list
    yaml_dict_list=[]

    #find centroids of objects
    labels=[]
    centroids=[]
    # TODO: Get the PointCloud for a given object and obtain it's centroid
    for object in object_list:
    	labels.append(object.label)
    	points_arr=ros_to_pcl(object.cloud)

    	center=np.mean(points_arr,axis=0)[:3]
    	centroids.append(center)
    # TODO: Get/Read parameters
    object_list_param = rospy.get_param('/object_list')#list with names and groups
    dropbox_param     = rospy.get_param('/dropbox')#name:left or right;group:green or red(dropbox);position [x,y,z]
    

    # TODO: Rotate PR2 in place to capture side tables for the collision map
        # Rotate Right
    pr2_robot.publish(-1.57)
    rospy.sleep(15.0)
    # Rotate Left
    pr2_robot.publish(1.57)
    rospy.sleep(30.0)
    # Rotate Center
    pr2_robot.publish(0)
    # TODO: Loop through the pick list
    for i in range(0,len(object_list_param)):
    	object_name.data= str(object_list_param[i]['name'])
    	object_group.data = str(object_list_param[i]['group'])
    	try:
    		index=labels.index(object_name.data)
    	except e:
    		print "not detected object"
    		continue

        pick_pose.position.x=np.asscalar(centroids[i][0])
        pick_pose.position.y=np.asscalar(centroids[i][1])
        pick_pose.position.z=np.asscalar(centroids[i][2])
    
    	position_drop=search_dict('group',object_group.data,'position',dropbox_param)
        # TODO: Create 'place_pose' for the object
        place_pose.position.x=position_drop[0]#from the list of position
        place_pose.position.y=position_drop[1]
        place_pose.position.z=position_drop[2]
        # TODO: Assign the arm to be used for pick_place
        arm_name.data=search_dict('group',object_group.data,'name',dropbox_param)
        # TODO: Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        dict_list=[]
        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')
        yaml_dict=make_yaml_dict(test_scene_num, object_name, arm_name, pick_pose, place_pose)
        yaml_dict_list.append(yaml_dict)
        print ("try for loop")
        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            # Insert your message variables to be sent as a service request
            print ("try for response")
            #resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)
            print ("Response: ",resp.success)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

    print ("try write")
    # TODO: Output your request parameters into output yaml file
    yaml_name="/home/robond/catkin_ws/src/RoboND-Perception-Project/pr2_robot/scripts/outputs_"+str(test_scene_num.data)+".yaml"
    send_to_yaml(yaml_name,yaml_dict_list)
    print ("success")
    return
if __name__ == '__main__':
    # TODO: ROS node initialization
	rospy.init_node('project_template',anonymous=True)
	# TODO: Create Subscribers
	pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)
	# TODO: Create Publishers
	pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
	pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
	pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
	pr2_robot=rospy.Publisher("/pr2/world_joint_controller/command",Float64,queue_size=1);
	pcl_test_pub = rospy.Publisher("/pcl_test", PointCloud2, queue_size=1)
	object_markers_pub=rospy.Publisher("/object_markers",Marker, queue_size=1)
	detected_objects_pub=rospy.Publisher("/detected_objects",DetectedObjectsArray, queue_size=1)
	# TODO: Load Model From disk
	model = pickle.load(open('model.sav', 'rb'))
	clf = model['classifier']
	encoder = LabelEncoder()
	encoder.classes_ = model['classes']
	scaler = model['scaler']
	# Initialize color_list
	get_color_list.color_list = []

	# TODO: Spin while node is not shutdown
	while not rospy.is_shutdown():
		rospy.spin()
