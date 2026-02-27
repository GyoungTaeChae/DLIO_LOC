/***********************************************************
 *                                                         *
 * Copyright (c)                                           *
 *                                                         *
 * The Verifiable & Control-Theoretic Robotics (VECTR) Lab *
 * University of California, Los Angeles                   *
 *                                                         *
 * Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   *
 * Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         *
 *                                                         *
 ***********************************************************/

#include "dlio/loc.h"

dlio::LocNode::LocNode(ros::NodeHandle node_handle) : nh(node_handle) {

  this->getParams();

  this->num_threads_ = omp_get_max_threads();

  this->loc_initialized = false;
  this->first_valid_scan = false;
  this->first_imu_received = false;
  if (this->imu_calibrate_) {this->imu_calibrated = false;}
  else {this->imu_calibrated = true;}
  this->deskew_status = false;
  this->deskew_size = 0;
  this->globalmap_ready = false;

  this->lidar_sub = this->nh.subscribe("pointcloud", 1,
      &dlio::LocNode::callbackPointCloud, this, ros::TransportHints().tcpNoDelay());
  this->imu_sub = this->nh.subscribe("imu", 1000,
      &dlio::LocNode::callbackImu, this, ros::TransportHints().tcpNoDelay());
  this->initialpose_sub = this->nh.subscribe("initialpose", 1,
      &dlio::LocNode::callbackInitialPose, this);

  this->odom_pub     = this->nh.advertise<nav_msgs::Odometry>("odom", 1, true);
  this->pose_pub     = this->nh.advertise<geometry_msgs::PoseStamped>("pose", 1, true);
  this->path_pub     = this->nh.advertise<nav_msgs::Path>("path", 1, true);
  this->deskewed_pub = this->nh.advertise<sensor_msgs::PointCloud2>("deskewed", 1, true);
  this->globalmap_pub = this->nh.advertise<sensor_msgs::PointCloud2>("globalmap", 1, true);

  this->publish_timer = this->nh.createTimer(ros::Duration(0.01), &dlio::LocNode::publishPose, this);

  this->T = Eigen::Matrix4f::Identity();
  this->T_prior = Eigen::Matrix4f::Identity();
  this->T_corr = Eigen::Matrix4f::Identity();

  this->origin = Eigen::Vector3f(0., 0., 0.);
  this->state.p = Eigen::Vector3f(0., 0., 0.);
  this->state.q = Eigen::Quaternionf(1., 0., 0., 0.);
  this->state.v.lin.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.lin.w = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.b = Eigen::Vector3f(0., 0., 0.);
  this->state.v.ang.w = Eigen::Vector3f(0., 0., 0.);

  this->lidarPose.p = Eigen::Vector3f(0., 0., 0.);
  this->lidarPose.q = Eigen::Quaternionf(1., 0., 0., 0.);

  this->imu_meas.stamp = 0.;
  this->imu_meas.ang_vel[0] = 0.;
  this->imu_meas.ang_vel[1] = 0.;
  this->imu_meas.ang_vel[2] = 0.;
  this->imu_meas.lin_accel[0] = 0.;
  this->imu_meas.lin_accel[1] = 0.;
  this->imu_meas.lin_accel[2] = 0.;

  this->imu_buffer.set_capacity(this->imu_buffer_size_);
  this->first_imu_stamp = 0.;
  this->prev_imu_stamp = 0.;

  this->original_scan = pcl::PointCloud<PointType>::ConstPtr (boost::make_shared<const pcl::PointCloud<PointType>>());
  this->deskewed_scan = pcl::PointCloud<PointType>::ConstPtr (boost::make_shared<const pcl::PointCloud<PointType>>());
  this->current_scan = pcl::PointCloud<PointType>::ConstPtr (boost::make_shared<const pcl::PointCloud<PointType>>());

  this->first_scan_stamp = 0.;
  this->elapsed_time = 0.;
  this->length_traversed = 0.;

  this->gicp.setCorrespondenceRandomness(this->gicp_k_correspondences_);
  this->gicp.setMaxCorrespondenceDistance(this->gicp_max_corr_dist_);
  this->gicp.setMaximumIterations(this->gicp_max_iter_);
  this->gicp.setTransformationEpsilon(this->gicp_transformation_ep_);
  this->gicp.setRotationEpsilon(this->gicp_rotation_ep_);
  this->gicp.setInitialLambdaFactor(this->gicp_init_lambda_factor_);

  pcl::Registration<PointType, PointType>::KdTreeReciprocalPtr temp;
  this->gicp.setSearchMethodSource(temp, true);
  this->gicp.setSearchMethodTarget(temp, true);

  this->geo.first_opt_done = false;
  this->geo.prev_vel = Eigen::Vector3f(0., 0., 0.);

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

  this->crop.setNegative(true);
  this->crop.setMin(Eigen::Vector4f(-this->crop_size_, -this->crop_size_, -this->crop_size_, 1.0));
  this->crop.setMax(Eigen::Vector4f(this->crop_size_, this->crop_size_, this->crop_size_, 1.0));

  this->voxel.setLeafSize(this->vf_res_, this->vf_res_, this->vf_res_);

  // Set initial pose
  if (this->use_yaml_) {
    Eigen::AngleAxisf rollAngle(this->init_roll_, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitchAngle(this->init_pitch_, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yawAngle(this->init_yaw_, Eigen::Vector3f::UnitZ());
    Eigen::Quaternionf q = yawAngle * pitchAngle * rollAngle;

    this->state.p = Eigen::Vector3f(this->init_x_, this->init_y_, this->init_z_);
    this->state.q = q;

    this->T.block(0, 0, 3, 3) = q.toRotationMatrix();
    this->T.block(0, 3, 3, 1) = this->state.p;

    this->lidarPose.p = this->state.p;
    this->lidarPose.q = this->state.q;

    this->geo.prev_p = this->state.p;
    this->geo.prev_q = this->state.q;

    ROS_INFO("Applied initial pose from yaml: position=[%.2f, %.2f, %.2f], quat=[w=%.4f, x=%.4f, y=%.4f, z=%.4f]",
             this->state.p[0], this->state.p[1], this->state.p[2],
             this->state.q.w(), this->state.q.x(), this->state.q.y(), this->state.q.z());
  } else if (this->use_rviz_) {
    ROS_INFO("Waiting for initial pose from RViz (/initialpose topic)");
  }

  // CPU Specs
  char CPUBrandString[0x40];
  memset(CPUBrandString, 0, sizeof(CPUBrandString));

  this->cpu_type = "";

  #ifdef HAS_CPUID
  unsigned int CPUInfo[4] = {0,0,0,0};
  __cpuid(0x80000000, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
  unsigned int nExIds = CPUInfo[0];
  for (unsigned int i = 0x80000000; i <= nExIds; ++i) {
    __cpuid(i, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
    if (i == 0x80000002)
      memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000003)
      memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
    else if (i == 0x80000004)
      memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
  }
  this->cpu_type = CPUBrandString;
  boost::trim(this->cpu_type);
  #endif

  FILE* file;
  struct tms timeSample;
  char line[128];

  this->lastCPU = times(&timeSample);
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;

  file = fopen("/proc/cpuinfo", "r");
  this->numProcessors = 0;
  while(fgets(line, 128, file) != nullptr) {
      if (strncmp(line, "processor", 9) == 0) this->numProcessors++;
  }
  fclose(file);

  // Load global map
  this->loadGlobalMap();

}

dlio::LocNode::~LocNode() {}

void dlio::LocNode::getParams() {

  // Version
  ros::param::param<std::string>("~dlio/version", this->version_, "0.0.0");

  // Frames
  ros::param::param<std::string>("~dlio/frames/map", this->map_frame, "map");
  ros::param::param<std::string>("~dlio/frames/baselink", this->baselink_frame, "base_link");
  ros::param::param<std::string>("~dlio/frames/lidar", this->lidar_frame, "lidar");
  ros::param::param<std::string>("~dlio/frames/imu", this->imu_frame, "imu");

  // Get Node NS and Remove Leading Character
  std::string ns = ros::this_node::getNamespace();
  ns.erase(0,1);

  // Concatenate Frame Name Strings
  this->map_frame = ns + "/" + this->map_frame;
  this->baselink_frame = ns + "/" + this->baselink_frame;
  this->lidar_frame = ns + "/" + this->lidar_frame;
  this->imu_frame = ns + "/" + this->imu_frame;

  // Deskew Flag
  ros::param::param<bool>("~dlio/pointcloud/deskew", this->deskew_, true);

  // Gravity
  ros::param::param<double>("~dlio/loc/gravity", this->gravity_, 9.80665);

  // Compute time offset between lidar and imu
  ros::param::param<bool>("~dlio/loc/computeTimeOffset", this->time_offset_, false);

  // Global Map
  ros::param::param<std::string>("~dlio/loc/globalmap/pcd_path", this->globalmap_pcd_path_, "");
  ros::param::param<double>("~dlio/loc/globalmap/leaf_size", this->globalmap_leaf_size_, 0.5);

  // Initial Pose
  ros::param::param<bool>("~dlio/loc/initial_pose/use_rviz", this->use_rviz_, false);
  ros::param::param<bool>("~dlio/loc/initial_pose/use_yaml", this->use_yaml_, true);
  ros::param::param<double>("~dlio/loc/initial_pose/x", this->init_x_, 0.0);
  ros::param::param<double>("~dlio/loc/initial_pose/y", this->init_y_, 0.0);
  ros::param::param<double>("~dlio/loc/initial_pose/z", this->init_z_, 0.0);
  ros::param::param<double>("~dlio/loc/initial_pose/roll", this->init_roll_, 0.0);
  ros::param::param<double>("~dlio/loc/initial_pose/pitch", this->init_pitch_, 0.0);
  ros::param::param<double>("~dlio/loc/initial_pose/yaw", this->init_yaw_, 0.0);

  ROS_INFO("Initial pose config: use_rviz=%s, use_yaml=%s",
           this->use_rviz_ ? "true" : "false", this->use_yaml_ ? "true" : "false");
  ROS_INFO("Initial pose yaml values: x=%.2f, y=%.2f, z=%.2f, roll=%.4f, pitch=%.4f, yaw=%.4f",
           this->init_x_, this->init_y_, this->init_z_,
           this->init_roll_, this->init_pitch_, this->init_yaw_);

  // Crop Box Filter
  ros::param::param<double>("~dlio/loc/preprocessing/cropBoxFilter/size", this->crop_size_, 1.0);

  // Voxel Grid Filter
  ros::param::param<bool>("~dlio/pointcloud/voxelize", this->vf_use_, true);
  ros::param::param<double>("~dlio/loc/preprocessing/voxelFilter/res", this->vf_res_, 0.05);

  // Extrinsics
  std::vector<float> t_default{0., 0., 0.};
  std::vector<float> R_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};

  // center of gravity to imu
  std::vector<float> baselink2imu_t, baselink2imu_R;
  ros::param::param<std::vector<float>>("~dlio/extrinsics/baselink2imu/t", baselink2imu_t, t_default);
  ros::param::param<std::vector<float>>("~dlio/extrinsics/baselink2imu/R", baselink2imu_R, R_default);
  this->extrinsics.baselink2imu.t =
    Eigen::Vector3f(baselink2imu_t[0], baselink2imu_t[1], baselink2imu_t[2]);
  this->extrinsics.baselink2imu.R =
    Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(baselink2imu_R.data(), 3, 3);

  this->extrinsics.baselink2imu_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2imu_T.block(0, 3, 3, 1) = this->extrinsics.baselink2imu.t;
  this->extrinsics.baselink2imu_T.block(0, 0, 3, 3) = this->extrinsics.baselink2imu.R;

  // center of gravity to lidar
  std::vector<float> baselink2lidar_t, baselink2lidar_R;
  ros::param::param<std::vector<float>>("~dlio/extrinsics/baselink2lidar/t", baselink2lidar_t, t_default);
  ros::param::param<std::vector<float>>("~dlio/extrinsics/baselink2lidar/R", baselink2lidar_R, R_default);

  this->extrinsics.baselink2lidar.t =
    Eigen::Vector3f(baselink2lidar_t[0], baselink2lidar_t[1], baselink2lidar_t[2]);
  this->extrinsics.baselink2lidar.R =
    Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(baselink2lidar_R.data(), 3, 3);

  this->extrinsics.baselink2lidar_T = Eigen::Matrix4f::Identity();
  this->extrinsics.baselink2lidar_T.block(0, 3, 3, 1) = this->extrinsics.baselink2lidar.t;
  this->extrinsics.baselink2lidar_T.block(0, 0, 3, 3) = this->extrinsics.baselink2lidar.R;

  // IMU
  ros::param::param<bool>("~dlio/loc/imu/calibration/accel", this->calibrate_accel_, true);
  ros::param::param<bool>("~dlio/loc/imu/calibration/gyro", this->calibrate_gyro_, true);
  ros::param::param<double>("~dlio/loc/imu/calibration/time", this->imu_calib_time_, 3.0);
  ros::param::param<int>("~dlio/loc/imu/bufferSize", this->imu_buffer_size_, 2000);

  std::vector<float> accel_default{0., 0., 0.}; std::vector<float> prior_accel_bias;
  std::vector<float> gyro_default{0., 0., 0.}; std::vector<float> prior_gyro_bias;

  ros::param::param<bool>("~dlio/loc/imu/approximateGravity", this->gravity_align_, true);
  ros::param::param<bool>("~dlio/imu/calibration", this->imu_calibrate_, true);
  ros::param::param<std::vector<float>>("~dlio/imu/intrinsics/accel/bias", prior_accel_bias, accel_default);
  ros::param::param<std::vector<float>>("~dlio/imu/intrinsics/gyro/bias", prior_gyro_bias, gyro_default);

  // scale-misalignment matrix
  std::vector<float> imu_sm_default{1., 0., 0., 0., 1., 0., 0., 0., 1.};
  std::vector<float> imu_sm;

  ros::param::param<std::vector<float>>("~dlio/imu/intrinsics/accel/sm", imu_sm, imu_sm_default);

  if (!this->imu_calibrate_) {
    this->state.b.accel[0] = prior_accel_bias[0];
    this->state.b.accel[1] = prior_accel_bias[1];
    this->state.b.accel[2] = prior_accel_bias[2];
    this->state.b.gyro[0] = prior_gyro_bias[0];
    this->state.b.gyro[1] = prior_gyro_bias[1];
    this->state.b.gyro[2] = prior_gyro_bias[2];
    this->imu_accel_sm_ = Eigen::Map<const Eigen::Matrix<float, -1, -1, Eigen::RowMajor>>(imu_sm.data(), 3, 3);
  } else {
    this->state.b.accel = Eigen::Vector3f(0., 0., 0.);
    this->state.b.gyro = Eigen::Vector3f(0., 0., 0.);
    this->imu_accel_sm_ = Eigen::Matrix3f::Identity();
  }

  // GICP
  ros::param::param<int>("~dlio/loc/gicp/minNumPoints", this->gicp_min_num_points_, 100);
  ros::param::param<int>("~dlio/loc/gicp/kCorrespondences", this->gicp_k_correspondences_, 20);
  ros::param::param<double>("~dlio/loc/gicp/maxCorrespondenceDistance", this->gicp_max_corr_dist_,
      std::sqrt(std::numeric_limits<double>::max()));
  ros::param::param<int>("~dlio/loc/gicp/maxIterations", this->gicp_max_iter_, 64);
  ros::param::param<double>("~dlio/loc/gicp/transformationEpsilon", this->gicp_transformation_ep_, 0.0005);
  ros::param::param<double>("~dlio/loc/gicp/rotationEpsilon", this->gicp_rotation_ep_, 0.0005);
  ros::param::param<double>("~dlio/loc/gicp/initLambdaFactor", this->gicp_init_lambda_factor_, 1e-9);

  // Geometric Observer
  ros::param::param<double>("~dlio/loc/geo/Kp", this->geo_Kp_, 1.0);
  ros::param::param<double>("~dlio/loc/geo/Kv", this->geo_Kv_, 1.0);
  ros::param::param<double>("~dlio/loc/geo/Kq", this->geo_Kq_, 1.0);
  ros::param::param<double>("~dlio/loc/geo/Kab", this->geo_Kab_, 1.0);
  ros::param::param<double>("~dlio/loc/geo/Kgb", this->geo_Kgb_, 1.0);
  ros::param::param<double>("~dlio/loc/geo/abias_max", this->geo_abias_max_, 1.0);
  ros::param::param<double>("~dlio/loc/geo/gbias_max", this->geo_gbias_max_, 1.0);

  ros::param::param<bool>("~dlio/verbose", this->verbose, true);
}

void dlio::LocNode::start() {
  if (!this->verbose) {
    return;
  }

  printf("\033[2J\033[1;1H");
  std::cout << std::endl
            << "+-------------------------------------------------------------------+" << std::endl;
  std::cout << "|          Direct LiDAR-Inertial Localization v" << this->version_  << "              |"
            << std::endl;
  std::cout << "+-------------------------------------------------------------------+" << std::endl;

}

void dlio::LocNode::loadGlobalMap() {

  if (this->globalmap_pcd_path_.empty()) {
    ROS_ERROR("Global map PCD path is empty!");
    return;
  }

  this->globalmap = pcl::PointCloud<PointType>::Ptr(boost::make_shared<pcl::PointCloud<PointType>>());

  if (pcl::io::loadPCDFile<PointType>(this->globalmap_pcd_path_, *this->globalmap) == -1) {
    ROS_ERROR("Failed to load global map from %s", this->globalmap_pcd_path_.c_str());
    return;
  }

  ROS_INFO("Loaded global map with %lu points from %s", this->globalmap->points.size(), this->globalmap_pcd_path_.c_str());

  // Downsample if leaf_size > 0
  if (this->globalmap_leaf_size_ > 0.0) {
    pcl::VoxelGrid<PointType> vg;
    vg.setLeafSize(this->globalmap_leaf_size_, this->globalmap_leaf_size_, this->globalmap_leaf_size_);
    vg.setInputCloud(this->globalmap);
    vg.filter(*this->globalmap);
    ROS_INFO("Downsampled global map to %lu points (leaf_size=%.2f)",
             this->globalmap->points.size(), this->globalmap_leaf_size_);
  }

  // Set as GICP target
  this->gicp.setInputTarget(this->globalmap);
  this->gicp.calculateTargetCovariances();

  this->globalmap_ready = true;
  ROS_INFO("Global map GICP target set. Ready for localization.");

  // Publish global map for RViz visualization
  sensor_msgs::PointCloud2 globalmap_ros;
  pcl::toROSMsg(*this->globalmap, globalmap_ros);
  globalmap_ros.header.stamp = ros::Time::now();
  globalmap_ros.header.frame_id = this->map_frame;
  this->globalmap_pub.publish(globalmap_ros);

}

void dlio::LocNode::callbackInitialPose(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg) {

  ROS_INFO("Received initial pose from topic");

  this->state.p[0] = msg->pose.pose.position.x;
  this->state.p[1] = msg->pose.pose.position.y;
  this->state.p[2] = msg->pose.pose.position.z;

  this->state.q.w() = msg->pose.pose.orientation.w;
  this->state.q.x() = msg->pose.pose.orientation.x;
  this->state.q.y() = msg->pose.pose.orientation.y;
  this->state.q.z() = msg->pose.pose.orientation.z;
  this->state.q.normalize();

  this->T.block(0, 0, 3, 3) = this->state.q.toRotationMatrix();
  this->T.block(0, 3, 3, 1) = this->state.p;
  this->T_prior = this->T;

  this->lidarPose.p = this->state.p;
  this->lidarPose.q = this->state.q;

  this->geo.prev_p = this->state.p;
  this->geo.prev_q = this->state.q;
  this->geo.prev_vel = Eigen::Vector3f(0., 0., 0.);

  this->state.v.lin.w = Eigen::Vector3f(0., 0., 0.);
  this->state.v.lin.b = Eigen::Vector3f(0., 0., 0.);

  ROS_INFO("Initial pose set to [%.2f, %.2f, %.2f]",
           this->state.p[0], this->state.p[1], this->state.p[2]);

}

void dlio::LocNode::publishPose(const ros::TimerEvent& e) {

  // nav_msgs::Odometry
  this->odom_ros.header.stamp = this->imu_stamp;
  this->odom_ros.header.frame_id = this->map_frame;
  this->odom_ros.child_frame_id = this->baselink_frame;

  this->odom_ros.pose.pose.position.x = this->state.p[0];
  this->odom_ros.pose.pose.position.y = this->state.p[1];
  this->odom_ros.pose.pose.position.z = this->state.p[2];

  this->odom_ros.pose.pose.orientation.w = this->state.q.w();
  this->odom_ros.pose.pose.orientation.x = this->state.q.x();
  this->odom_ros.pose.pose.orientation.y = this->state.q.y();
  this->odom_ros.pose.pose.orientation.z = this->state.q.z();

  this->odom_ros.twist.twist.linear.x = this->state.v.lin.w[0];
  this->odom_ros.twist.twist.linear.y = this->state.v.lin.w[1];
  this->odom_ros.twist.twist.linear.z = this->state.v.lin.w[2];

  this->odom_ros.twist.twist.angular.x = this->state.v.ang.b[0];
  this->odom_ros.twist.twist.angular.y = this->state.v.ang.b[1];
  this->odom_ros.twist.twist.angular.z = this->state.v.ang.b[2];

  this->odom_pub.publish(this->odom_ros);

  // geometry_msgs::PoseStamped
  this->pose_ros.header.stamp = this->imu_stamp;
  this->pose_ros.header.frame_id = this->map_frame;

  this->pose_ros.pose.position.x = this->state.p[0];
  this->pose_ros.pose.position.y = this->state.p[1];
  this->pose_ros.pose.position.z = this->state.p[2];

  this->pose_ros.pose.orientation.w = this->state.q.w();
  this->pose_ros.pose.orientation.x = this->state.q.x();
  this->pose_ros.pose.orientation.y = this->state.q.y();
  this->pose_ros.pose.orientation.z = this->state.q.z();

  this->pose_pub.publish(this->pose_ros);

}

void dlio::LocNode::publishToROS(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud) {
  this->publishCloud(published_cloud, T_cloud);

  // nav_msgs::Path
  this->path_ros.header.stamp = this->imu_stamp;
  this->path_ros.header.frame_id = this->map_frame;

  geometry_msgs::PoseStamped p;
  p.header.stamp = this->imu_stamp;
  p.header.frame_id = this->map_frame;
  p.pose.position.x = this->state.p[0];
  p.pose.position.y = this->state.p[1];
  p.pose.position.z = this->state.p[2];
  p.pose.orientation.w = this->state.q.w();
  p.pose.orientation.x = this->state.q.x();
  p.pose.orientation.y = this->state.q.y();
  p.pose.orientation.z = this->state.q.z();

  this->path_ros.poses.push_back(p);
  this->path_pub.publish(this->path_ros);

  // transform: map to baselink
  static tf2_ros::TransformBroadcaster br;
  geometry_msgs::TransformStamped transformStamped;

  transformStamped.header.stamp = this->imu_stamp;
  transformStamped.header.frame_id = this->map_frame;
  transformStamped.child_frame_id = this->baselink_frame;

  transformStamped.transform.translation.x = this->state.p[0];
  transformStamped.transform.translation.y = this->state.p[1];
  transformStamped.transform.translation.z = this->state.p[2];

  transformStamped.transform.rotation.w = this->state.q.w();
  transformStamped.transform.rotation.x = this->state.q.x();
  transformStamped.transform.rotation.y = this->state.q.y();
  transformStamped.transform.rotation.z = this->state.q.z();

  br.sendTransform(transformStamped);

  // transform: baselink to imu
  transformStamped.header.stamp = this->imu_stamp;
  transformStamped.header.frame_id = this->baselink_frame;
  transformStamped.child_frame_id = this->imu_frame;

  transformStamped.transform.translation.x = this->extrinsics.baselink2imu.t[0];
  transformStamped.transform.translation.y = this->extrinsics.baselink2imu.t[1];
  transformStamped.transform.translation.z = this->extrinsics.baselink2imu.t[2];

  Eigen::Quaternionf q(this->extrinsics.baselink2imu.R);
  transformStamped.transform.rotation.w = q.w();
  transformStamped.transform.rotation.x = q.x();
  transformStamped.transform.rotation.y = q.y();
  transformStamped.transform.rotation.z = q.z();

  br.sendTransform(transformStamped);

  // transform: baselink to lidar
  transformStamped.header.stamp = this->imu_stamp;
  transformStamped.header.frame_id = this->baselink_frame;
  transformStamped.child_frame_id = this->lidar_frame;

  transformStamped.transform.translation.x = this->extrinsics.baselink2lidar.t[0];
  transformStamped.transform.translation.y = this->extrinsics.baselink2lidar.t[1];
  transformStamped.transform.translation.z = this->extrinsics.baselink2lidar.t[2];

  Eigen::Quaternionf qq(this->extrinsics.baselink2lidar.R);
  transformStamped.transform.rotation.w = qq.w();
  transformStamped.transform.rotation.x = qq.x();
  transformStamped.transform.rotation.y = qq.y();
  transformStamped.transform.rotation.z = qq.z();

  br.sendTransform(transformStamped);

}

void dlio::LocNode::publishCloud(pcl::PointCloud<PointType>::ConstPtr published_cloud, Eigen::Matrix4f T_cloud) {

  pcl::PointCloud<PointType>::Ptr deskewed_scan_t_ (boost::make_shared<pcl::PointCloud<PointType>>());

  pcl::transformPointCloud (*published_cloud, *deskewed_scan_t_, T_cloud);

  // published deskewed cloud
  sensor_msgs::PointCloud2 deskewed_ros;
  pcl::toROSMsg(*deskewed_scan_t_, deskewed_ros);
  deskewed_ros.header.stamp = this->scan_header_stamp;
  deskewed_ros.header.frame_id = this->map_frame;
  this->deskewed_pub.publish(deskewed_ros);

}

void dlio::LocNode::getScanFromROS(const sensor_msgs::PointCloud2ConstPtr& pc) {

  pcl::PointCloud<PointType>::Ptr original_scan_ (boost::make_shared<pcl::PointCloud<PointType>>());
  pcl::fromROSMsg(*pc, *original_scan_);

  // Remove NaNs
  std::vector<int> idx;
  original_scan_->is_dense = false;
  pcl::removeNaNFromPointCloud(*original_scan_, *original_scan_, idx);

  // Crop Box Filter
  this->crop.setInputCloud(original_scan_);
  this->crop.filter(*original_scan_);

  // automatically detect sensor type
  this->sensor = dlio::SensorType::UNKNOWN;
  for (auto &field : pc->fields) {
    if (field.name == "t") {
      this->sensor = dlio::SensorType::OUSTER;
      break;
    } else if (field.name == "time") {
      this->sensor = dlio::SensorType::VELODYNE;
      break;
    } else if (field.name == "timestamp" && original_scan_->points[0].timestamp < 1e14) {
      this->sensor = dlio::SensorType::HESAI;
      break;
    } else if (field.name == "timestamp" && original_scan_->points[0].timestamp > 1e14) {
      this->sensor = dlio::SensorType::LIVOX;
      break;
    }
  }

  if (this->sensor == dlio::SensorType::UNKNOWN) {
    this->deskew_ = false;
  }

  this->scan_header_stamp = pc->header.stamp;
  this->original_scan = original_scan_;

}

void dlio::LocNode::preprocessPoints() {

  // Deskew the original dlio-type scan
  if (this->deskew_) {

    this->deskewPointcloud();

    if (!this->first_valid_scan) {
      return;
    }

  } else {

    this->scan_stamp = this->scan_header_stamp.toSec();

    // don't process scans until IMU data is present
    if (!this->first_valid_scan) {

      if (this->imu_buffer.empty() || this->scan_stamp <= this->imu_buffer.back().stamp) {
        return;
      }

      this->first_valid_scan = true;
      this->T_prior = this->T; // assume no motion for the first scan

    } else {

      // IMU prior for second scan onwards
    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> frames;
      frames = this->integrateImu(this->prev_scan_stamp, this->lidarPose.q, this->lidarPose.p,
                                this->geo.prev_vel.cast<float>(), {this->scan_stamp});

    if (frames.size() > 0) {
      this->T_prior = frames.back();
    } else {
      this->T_prior = this->T;
    }

    }

    pcl::PointCloud<PointType>::Ptr deskewed_scan_ (boost::make_shared<pcl::PointCloud<PointType>>());
    pcl::transformPointCloud (*this->original_scan, *deskewed_scan_,
                              this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = false;
  }

  // Voxel Grid Filter
  if (this->vf_use_) {
    pcl::PointCloud<PointType>::Ptr current_scan_
      (boost::make_shared<pcl::PointCloud<PointType>>(*this->deskewed_scan));
    this->voxel.setInputCloud(current_scan_);
    this->voxel.filter(*current_scan_);
    this->current_scan = current_scan_;
  } else {
    this->current_scan = this->deskewed_scan;
  }

}

void dlio::LocNode::deskewPointcloud() {

  pcl::PointCloud<PointType>::Ptr deskewed_scan_ (boost::make_shared<pcl::PointCloud<PointType>>());
  deskewed_scan_->points.resize(this->original_scan->points.size());

  // individual point timestamps should be relative to this time
  double sweep_ref_time = this->scan_header_stamp.toSec();

  // sort points by timestamp and build list of timestamps
  std::function<bool(const PointType&, const PointType&)> point_time_cmp;
  std::function<bool(boost::range::index_value<PointType&, long>,
                     boost::range::index_value<PointType&, long>)> point_time_neq;
  std::function<double(boost::range::index_value<PointType&, long>)> extract_point_time;

  if (this->sensor == dlio::SensorType::OUSTER) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.t < p2.t; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().t != p2.value().t; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return sweep_ref_time + pt.value().t * 1e-9f; };

  } else if (this->sensor == dlio::SensorType::VELODYNE) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.time < p2.time; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().time != p2.value().time; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return sweep_ref_time + pt.value().time; };

  } else if (this->sensor == dlio::SensorType::HESAI) {

    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.timestamp < p2.timestamp; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().timestamp != p2.value().timestamp; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return pt.value().timestamp; };

  } else if (this->sensor == dlio::SensorType::LIVOX) {
    point_time_cmp = [](const PointType& p1, const PointType& p2)
      { return p1.timestamp < p2.timestamp; };
    point_time_neq = [](boost::range::index_value<PointType&, long> p1,
                        boost::range::index_value<PointType&, long> p2)
      { return p1.value().timestamp != p2.value().timestamp; };
    extract_point_time = [&sweep_ref_time](boost::range::index_value<PointType&, long> pt)
      { return pt.value().timestamp * 1e-9f; };
  }

  // copy points into deskewed_scan_ in order of timestamp
  std::partial_sort_copy(this->original_scan->points.begin(), this->original_scan->points.end(),
                         deskewed_scan_->points.begin(), deskewed_scan_->points.end(), point_time_cmp);

  // filter unique timestamps
  auto points_unique_timestamps = deskewed_scan_->points
                                  | boost::adaptors::indexed()
                                  | boost::adaptors::adjacent_filtered(point_time_neq);

  // extract timestamps from points and put them in their own list
  std::vector<double> timestamps;
  std::vector<int> unique_time_indices;

  // compute offset between sweep reference time and first point timestamp
  double offset = 0.0;
  if (this->time_offset_) {
    offset = sweep_ref_time - extract_point_time(*points_unique_timestamps.begin());
  }

  // build list of unique timestamps and indices of first point with each timestamp
  for (auto it = points_unique_timestamps.begin(); it != points_unique_timestamps.end(); it++) {
    timestamps.push_back(extract_point_time(*it) + offset);
    unique_time_indices.push_back(it->index());
  }
  unique_time_indices.push_back(deskewed_scan_->points.size());

  int median_pt_index = timestamps.size() / 2;
  this->scan_stamp = timestamps[median_pt_index]; // set this->scan_stamp to the timestamp of the median point

  // don't process scans until IMU data is present
  if (!this->first_valid_scan) {
    if (this->imu_buffer.empty() || this->scan_stamp <= this->imu_buffer.back().stamp) {
      return;
    }

    this->first_valid_scan = true;
    this->T_prior = this->T; // assume no motion for the first scan
    pcl::transformPointCloud (*deskewed_scan_, *deskewed_scan_, this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = true;
    return;
  }

  // IMU prior & deskewing for second scan onwards
  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> frames;
  frames = this->integrateImu(this->prev_scan_stamp, this->lidarPose.q, this->lidarPose.p,
                              this->geo.prev_vel.cast<float>(), timestamps);
  this->deskew_size = frames.size(); // if integration successful, equal to timestamps.size()

  // if there are no frames between the start and end of the sweep
  // that probably means that there's a sync issue
  if (frames.size() != timestamps.size()) {
    ROS_FATAL("Bad time sync between LiDAR and IMU!");

    this->T_prior = this->T;
    pcl::transformPointCloud (*deskewed_scan_, *deskewed_scan_, this->T_prior * this->extrinsics.baselink2lidar_T);
    this->deskewed_scan = deskewed_scan_;
    this->deskew_status = false;
    return;
  }

  // update prior to be the estimated pose at the median time of the scan (corresponds to this->scan_stamp)
  this->T_prior = frames[median_pt_index];

#pragma omp parallel for num_threads(this->num_threads_)
  for (int i = 0; i < timestamps.size(); i++) {

    Eigen::Matrix4f T = frames[i] * this->extrinsics.baselink2lidar_T;

    // transform point to world frame
    for (int k = unique_time_indices[i]; k < unique_time_indices[i+1]; k++) {
      auto &pt = deskewed_scan_->points[k];
      pt.getVector4fMap()[3] = 1.;
      pt.getVector4fMap() = T * pt.getVector4fMap();
    }
  }

  this->deskewed_scan = deskewed_scan_;
  this->deskew_status = true;

}

void dlio::LocNode::setInputSource() {
  this->gicp.setInputSource(this->current_scan);
  this->gicp.calculateSourceCovariances();
}

void dlio::LocNode::initializeLoc() {

  // Wait for IMU
  if (!this->first_imu_received || !this->imu_calibrated) {
    return;
  }

  // Wait for global map
  if (!this->globalmap_ready) {
    return;
  }

  this->loc_initialized = true;
  std::cout << std::endl << " DLIO Localization initialized!" << std::endl;

}

void dlio::LocNode::callbackPointCloud(const sensor_msgs::PointCloud2ConstPtr& pc) {

  double then = ros::Time::now().toSec();

  if (this->first_scan_stamp == 0.) {
    this->first_scan_stamp = pc->header.stamp.toSec();
  }

  // Localization Initialization procedures (IMU calib, global map check)
  if (!this->loc_initialized) {
    this->initializeLoc();
  }

  // Convert incoming scan into DLIO format
  this->getScanFromROS(pc);

  // Preprocess points
  this->preprocessPoints();

  if (!this->first_valid_scan) {
    return;
  }

  if (this->current_scan->points.size() <= this->gicp_min_num_points_) {
    ROS_FATAL("Low number of points in the cloud!");
    return;
  }

  // Set new frame as input source
  this->setInputSource();

  // First scan: just set initial state, no alignment needed
  if (!this->geo.first_opt_done) {
    this->prev_scan_stamp = this->scan_stamp;
    this->geo.first_opt_done = true;
    return;
  }

  // Get the next pose via IMU + S2M + GEO
  this->getNextPose();

  // Update trajectory
  this->trajectory.push_back( std::make_pair(this->state.p, this->state.q) );

  // Update time stamps
  this->lidar_rates.push_back( 1. / (this->scan_stamp - this->prev_scan_stamp) );
  this->prev_scan_stamp = this->scan_stamp;
  this->elapsed_time = this->scan_stamp - this->first_scan_stamp;

  // Publish stuff to ROS
  pcl::PointCloud<PointType>::ConstPtr published_cloud = this->current_scan;
  this->publish_thread = std::thread( &dlio::LocNode::publishToROS, this, published_cloud, this->T_corr );
  this->publish_thread.detach();

  // Update some statistics
  this->comp_times.push_back(ros::Time::now().toSec() - then);
  this->gicp_hasConverged = this->gicp.hasConverged();

  // Debug statements
  if (this->verbose) {
    this->debug_thread = std::thread( &dlio::LocNode::debug, this );
    this->debug_thread.detach();
  }
}

void dlio::LocNode::callbackImu(const sensor_msgs::Imu::ConstPtr& imu_raw) {

  this->first_imu_received = true;

  sensor_msgs::Imu::Ptr imu = this->transformImu( imu_raw );
  this->imu_stamp = imu->header.stamp;

  Eigen::Vector3f lin_accel;
  Eigen::Vector3f ang_vel;

  // Get IMU samples
  ang_vel[0] = imu->angular_velocity.x;
  ang_vel[1] = imu->angular_velocity.y;
  ang_vel[2] = imu->angular_velocity.z;

  lin_accel[0] = imu->linear_acceleration.x;
  lin_accel[1] = imu->linear_acceleration.y;
  lin_accel[2] = imu->linear_acceleration.z;

  if (this->first_imu_stamp == 0.) {
    this->first_imu_stamp = imu->header.stamp.toSec();
  }

  // IMU calibration procedure - do for three seconds
  if (!this->imu_calibrated) {

    static int num_samples = 0;
    static Eigen::Vector3f gyro_avg (0., 0., 0.);
    static Eigen::Vector3f accel_avg (0., 0., 0.);
    static bool print = true;

    if ((imu->header.stamp.toSec() - this->first_imu_stamp) < this->imu_calib_time_) {

      num_samples++;

      gyro_avg[0] += ang_vel[0];
      gyro_avg[1] += ang_vel[1];
      gyro_avg[2] += ang_vel[2];

      accel_avg[0] += lin_accel[0];
      accel_avg[1] += lin_accel[1];
      accel_avg[2] += lin_accel[2];

      if(print) {
        std::cout << std::endl << " Calibrating IMU for " << this->imu_calib_time_ << " seconds... ";
        std::cout.flush();
        print = false;
      }

    } else {

      std::cout << "done" << std::endl << std::endl;

      gyro_avg /= num_samples;
      accel_avg /= num_samples;

      Eigen::Vector3f grav_vec (0., 0., this->gravity_);

      if (this->gravity_align_ && !this->use_yaml_) {

        // Estimate gravity vector - Only approximate if biases have not been pre-calibrated
        grav_vec = (accel_avg - this->state.b.accel).normalized() * abs(this->gravity_);
        Eigen::Quaternionf grav_q = Eigen::Quaternionf::FromTwoVectors(grav_vec, Eigen::Vector3f(0., 0., this->gravity_));

        // use_yaml이면 이미 초기 pose가 설정되어 있으므로 gravity alignment 스킵
        this->state.q = grav_q;
        this->T.block(0,0,3,3) = this->state.q.toRotationMatrix();
        this->lidarPose.q = this->state.q;

        // rpy
        auto euler = grav_q.toRotationMatrix().eulerAngles(2, 1, 0);
        double yaw = euler[0] * (180.0/M_PI);
        double pitch = euler[1] * (180.0/M_PI);
        double roll = euler[2] * (180.0/M_PI);

        // use alternate representation if the yaw is smaller
        if (abs(remainder(yaw + 180.0, 360.0)) < abs(yaw)) {
          yaw   = remainder(yaw + 180.0,   360.0);
          pitch = remainder(180.0 - pitch, 360.0);
          roll  = remainder(roll + 180.0,  360.0);
        }
        std::cout << " Estimated initial attitude:" << std::endl;
        std::cout << "   Roll  [deg]: " << to_string_with_precision(roll, 4) << std::endl;
        std::cout << "   Pitch [deg]: " << to_string_with_precision(pitch, 4) << std::endl;
        std::cout << "   Yaw   [deg]: " << to_string_with_precision(yaw, 4) << std::endl;
        std::cout << std::endl;
      } else if (this->use_yaml_) {
        ROS_INFO("Skipping gravity alignment (using yaml initial pose)");
      }

      if (this->calibrate_accel_) {

        // subtract gravity from avg accel to get bias
        this->state.b.accel = accel_avg - grav_vec;

        std::cout << " Accel biases [xyz]: " << to_string_with_precision(this->state.b.accel[0], 8) << ", "
                                             << to_string_with_precision(this->state.b.accel[1], 8) << ", "
                                             << to_string_with_precision(this->state.b.accel[2], 8) << std::endl;
      }

      if (this->calibrate_gyro_) {

        this->state.b.gyro = gyro_avg;

        std::cout << " Gyro biases  [xyz]: " << to_string_with_precision(this->state.b.gyro[0], 8) << ", "
                                             << to_string_with_precision(this->state.b.gyro[1], 8) << ", "
                                             << to_string_with_precision(this->state.b.gyro[2], 8) << std::endl;
      }

      this->imu_calibrated = true;

    }

  } else {

    double dt = imu->header.stamp.toSec() - this->prev_imu_stamp;
    if (dt == 0) { dt = 1.0/200.0; }
    this->imu_rates.push_back( 1./dt );

    // Apply the calibrated bias to the new IMU measurements
    this->imu_meas.stamp = imu->header.stamp.toSec();
    this->imu_meas.dt = dt;
    this->prev_imu_stamp = this->imu_meas.stamp;

    Eigen::Vector3f lin_accel_corrected = (this->imu_accel_sm_ * lin_accel) - this->state.b.accel;
    Eigen::Vector3f ang_vel_corrected = ang_vel - this->state.b.gyro;

    this->imu_meas.lin_accel = lin_accel_corrected;
    this->imu_meas.ang_vel = ang_vel_corrected;

    // Store calibrated IMU measurements into imu buffer for manual integration later.
    this->mtx_imu.lock();
    this->imu_buffer.push_front(this->imu_meas);
    this->mtx_imu.unlock();

    // Notify the callbackPointCloud thread that IMU data exists for this time
    this->cv_imu_stamp.notify_one();

    if (this->geo.first_opt_done) {
      // Geometric Observer: Propagate State
      this->propagateState();
    }

  }

}

void dlio::LocNode::getNextPose() {

  // Global map is already set as target in loadGlobalMap()
  // No submap check/swap needed

  // Align with global map with global IMU transformation as initial guess
  pcl::PointCloud<PointType>::Ptr aligned (boost::make_shared<pcl::PointCloud<PointType>>());
  double gicp_then = ros::Time::now().toSec();
  this->gicp.align(*aligned);
  this->gicp_times.push_back(ros::Time::now().toSec() - gicp_then);

  // Get final transformation in global frame
  this->T_corr = this->gicp.getFinalTransformation(); // "correction" transformation
  this->T = this->T_corr * this->T_prior;

  // Update next global pose
  this->propagateGICP();

  // Geometric observer update
  this->updateState();

}

bool dlio::LocNode::imuMeasFromTimeRange(double start_time, double end_time,
                                          boost::circular_buffer<ImuMeas>::reverse_iterator& begin_imu_it,
                                          boost::circular_buffer<ImuMeas>::reverse_iterator& end_imu_it) {

  if (this->imu_buffer.empty() || this->imu_buffer.front().stamp < end_time) {
    // Wait for the latest IMU data
    std::unique_lock<decltype(this->mtx_imu)> lock(this->mtx_imu);
    this->cv_imu_stamp.wait(lock, [this, &end_time]{ return this->imu_buffer.front().stamp >= end_time; });
  }

  auto imu_it = this->imu_buffer.begin();

  auto last_imu_it = imu_it;
  imu_it++;
  while (imu_it != this->imu_buffer.end() && imu_it->stamp >= end_time) {
    last_imu_it = imu_it;
    imu_it++;
  }

  while (imu_it != this->imu_buffer.end() && imu_it->stamp >= start_time) {
    imu_it++;
  }

  if (imu_it == this->imu_buffer.end()) {
    // not enough IMU measurements, return false
    return false;
  }
  imu_it++;

  // Set reverse iterators (to iterate forward in time)
  end_imu_it = boost::circular_buffer<ImuMeas>::reverse_iterator(last_imu_it);
  begin_imu_it = boost::circular_buffer<ImuMeas>::reverse_iterator(imu_it);

  return true;
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
dlio::LocNode::integrateImu(double start_time, Eigen::Quaternionf q_init, Eigen::Vector3f p_init,
                             Eigen::Vector3f v_init, const std::vector<double>& sorted_timestamps) {

  const std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> empty;

  if (sorted_timestamps.empty() || start_time > sorted_timestamps.front()) {
    // invalid input, return empty vector
    return empty;
  }

  boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it;
  boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it;
  if (this->imuMeasFromTimeRange(start_time, sorted_timestamps.back(), begin_imu_it, end_imu_it) == false) {
    // not enough IMU measurements, return empty vector
    return empty;
  }

  // Backwards integration to find pose at first IMU sample
  const ImuMeas& f1 = *begin_imu_it;
  const ImuMeas& f2 = *(begin_imu_it+1);

  // Time between first two IMU samples
  double dt = f2.dt;

  // Time between first IMU sample and start_time
  double idt = start_time - f1.stamp;

  // Angular acceleration between first two IMU samples
  Eigen::Vector3f alpha_dt = f2.ang_vel - f1.ang_vel;
  Eigen::Vector3f alpha = alpha_dt / dt;

  // Average angular velocity (reversed) between first IMU sample and start_time
  Eigen::Vector3f omega_i = -(f1.ang_vel + 0.5*alpha*idt);

  // Set q_init to orientation at first IMU sample
  q_init = Eigen::Quaternionf (
    q_init.w() - 0.5*( q_init.x()*omega_i[0] + q_init.y()*omega_i[1] + q_init.z()*omega_i[2] ) * idt,
    q_init.x() + 0.5*( q_init.w()*omega_i[0] - q_init.z()*omega_i[1] + q_init.y()*omega_i[2] ) * idt,
    q_init.y() + 0.5*( q_init.z()*omega_i[0] + q_init.w()*omega_i[1] - q_init.x()*omega_i[2] ) * idt,
    q_init.z() + 0.5*( q_init.x()*omega_i[1] - q_init.y()*omega_i[0] + q_init.w()*omega_i[2] ) * idt
  );
  q_init.normalize();

  // Average angular velocity between first two IMU samples
  Eigen::Vector3f omega = f1.ang_vel + 0.5*alpha_dt;

  // Orientation at second IMU sample
  Eigen::Quaternionf q2 (
    q_init.w() - 0.5*( q_init.x()*omega[0] + q_init.y()*omega[1] + q_init.z()*omega[2] ) * dt,
    q_init.x() + 0.5*( q_init.w()*omega[0] - q_init.z()*omega[1] + q_init.y()*omega[2] ) * dt,
    q_init.y() + 0.5*( q_init.z()*omega[0] + q_init.w()*omega[1] - q_init.x()*omega[2] ) * dt,
    q_init.z() + 0.5*( q_init.x()*omega[1] - q_init.y()*omega[0] + q_init.w()*omega[2] ) * dt
  );
  q2.normalize();

  // Acceleration at first IMU sample
  Eigen::Vector3f a1 = q_init._transformVector(f1.lin_accel);
  a1[2] -= this->gravity_;

  // Acceleration at second IMU sample
  Eigen::Vector3f a2 = q2._transformVector(f2.lin_accel);
  a2[2] -= this->gravity_;

  // Jerk between first two IMU samples
  Eigen::Vector3f j = (a2 - a1) / dt;

  // Set v_init to velocity at first IMU sample (go backwards from start_time)
  v_init -= a1*idt + 0.5*j*idt*idt;

  // Set p_init to position at first IMU sample (go backwards from start_time)
  p_init -= v_init*idt + 0.5*a1*idt*idt + (1/6.)*j*idt*idt*idt;

  return this->integrateImuInternal(q_init, p_init, v_init, sorted_timestamps, begin_imu_it, end_imu_it);
}

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>>
dlio::LocNode::integrateImuInternal(Eigen::Quaternionf q_init, Eigen::Vector3f p_init, Eigen::Vector3f v_init,
                                     const std::vector<double>& sorted_timestamps,
                                     boost::circular_buffer<ImuMeas>::reverse_iterator begin_imu_it,
                                     boost::circular_buffer<ImuMeas>::reverse_iterator end_imu_it) {

  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f>> imu_se3;

  // Initialization
  Eigen::Quaternionf q = q_init;
  Eigen::Vector3f p = p_init;
  Eigen::Vector3f v = v_init;
  Eigen::Vector3f a = q._transformVector(begin_imu_it->lin_accel);
  a[2] -= this->gravity_;

  // Iterate over IMU measurements and timestamps
  auto prev_imu_it = begin_imu_it;
  auto imu_it = prev_imu_it + 1;

  auto stamp_it = sorted_timestamps.begin();

  for (; imu_it != end_imu_it; imu_it++) {

    const ImuMeas& f0 = *prev_imu_it;
    const ImuMeas& f = *imu_it;

    // Time between IMU samples
    double dt = f.dt;

    // Angular acceleration
    Eigen::Vector3f alpha_dt = f.ang_vel - f0.ang_vel;
    Eigen::Vector3f alpha = alpha_dt / dt;

    // Average angular velocity
    Eigen::Vector3f omega = f0.ang_vel + 0.5*alpha_dt;

    // Orientation
    q = Eigen::Quaternionf (
      q.w() - 0.5*( q.x()*omega[0] + q.y()*omega[1] + q.z()*omega[2] ) * dt,
      q.x() + 0.5*( q.w()*omega[0] - q.z()*omega[1] + q.y()*omega[2] ) * dt,
      q.y() + 0.5*( q.z()*omega[0] + q.w()*omega[1] - q.x()*omega[2] ) * dt,
      q.z() + 0.5*( q.x()*omega[1] - q.y()*omega[0] + q.w()*omega[2] ) * dt
    );
    q.normalize();

    // Acceleration
    Eigen::Vector3f a0 = a;
    a = q._transformVector(f.lin_accel);
    a[2] -= this->gravity_;

    // Jerk
    Eigen::Vector3f j_dt = a - a0;
    Eigen::Vector3f j = j_dt / dt;

    // Interpolate for given timestamps
    while (stamp_it != sorted_timestamps.end() && *stamp_it <= f.stamp) {
      // Time between previous IMU sample and given timestamp
      double idt = *stamp_it - f0.stamp;

      // Average angular velocity
      Eigen::Vector3f omega_i = f0.ang_vel + 0.5*alpha*idt;

      // Orientation
      Eigen::Quaternionf q_i (
        q.w() - 0.5*( q.x()*omega_i[0] + q.y()*omega_i[1] + q.z()*omega_i[2] ) * idt,
        q.x() + 0.5*( q.w()*omega_i[0] - q.z()*omega_i[1] + q.y()*omega_i[2] ) * idt,
        q.y() + 0.5*( q.z()*omega_i[0] + q.w()*omega_i[1] - q.x()*omega_i[2] ) * idt,
        q.z() + 0.5*( q.x()*omega_i[1] - q.y()*omega_i[0] + q.w()*omega_i[2] ) * idt
      );
      q_i.normalize();

      // Position
      Eigen::Vector3f p_i = p + v*idt + 0.5*a0*idt*idt + (1/6.)*j*idt*idt*idt;

      // Transformation
      Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
      T.block(0, 0, 3, 3) = q_i.toRotationMatrix();
      T.block(0, 3, 3, 1) = p_i;

      imu_se3.push_back(T);

      stamp_it++;
    }

    // Position
    p += v*dt + 0.5*a0*dt*dt + (1/6.)*j_dt*dt*dt;

    // Velocity
    v += a0*dt + 0.5*j_dt*dt;

    prev_imu_it = imu_it;

  }

  return imu_se3;

}

void dlio::LocNode::propagateGICP() {

  this->lidarPose.p << this->T(0,3), this->T(1,3), this->T(2,3);

  Eigen::Matrix3f rotSO3;
  rotSO3 << this->T(0,0), this->T(0,1), this->T(0,2),
            this->T(1,0), this->T(1,1), this->T(1,2),
            this->T(2,0), this->T(2,1), this->T(2,2);

  Eigen::Quaternionf q(rotSO3);

  // Normalize quaternion
  double norm = sqrt(q.w()*q.w() + q.x()*q.x() + q.y()*q.y() + q.z()*q.z());
  q.w() /= norm; q.x() /= norm; q.y() /= norm; q.z() /= norm;
  this->lidarPose.q = q;

}

void dlio::LocNode::propagateState() {

  // Lock thread to prevent state from being accessed by UpdateState
  std::lock_guard<std::mutex> lock( this->geo.mtx );

  double dt = this->imu_meas.dt;

  Eigen::Quaternionf qhat = this->state.q, omega;
  Eigen::Vector3f world_accel;

  // Transform accel from body to world frame
  world_accel = qhat._transformVector(this->imu_meas.lin_accel);

  // Accel propogation
  this->state.p[0] += this->state.v.lin.w[0]*dt + 0.5*dt*dt*world_accel[0];
  this->state.p[1] += this->state.v.lin.w[1]*dt + 0.5*dt*dt*world_accel[1];
  this->state.p[2] += this->state.v.lin.w[2]*dt + 0.5*dt*dt*(world_accel[2] - this->gravity_);

  this->state.v.lin.w[0] += world_accel[0]*dt;
  this->state.v.lin.w[1] += world_accel[1]*dt;
  this->state.v.lin.w[2] += (world_accel[2] - this->gravity_)*dt;
  this->state.v.lin.b = this->state.q.toRotationMatrix().inverse() * this->state.v.lin.w;

  // Gyro propogation
  omega.w() = 0;
  omega.vec() = this->imu_meas.ang_vel;
  Eigen::Quaternionf tmp = qhat * omega;
  this->state.q.w() += 0.5 * dt * tmp.w();
  this->state.q.vec() += 0.5 * dt * tmp.vec();

  // Ensure quaternion is properly normalized
  this->state.q.normalize();

  this->state.v.ang.b = this->imu_meas.ang_vel;
  this->state.v.ang.w = this->state.q.toRotationMatrix() * this->state.v.ang.b;

}

void dlio::LocNode::updateState() {

  // Lock thread to prevent state from being accessed by PropagateState
  std::lock_guard<std::mutex> lock( this->geo.mtx );

  Eigen::Vector3f pin = this->lidarPose.p;
  Eigen::Quaternionf qin = this->lidarPose.q;
  double dt = this->scan_stamp - this->prev_scan_stamp;

  Eigen::Quaternionf qe, qhat, qcorr;
  qhat = this->state.q;

  // Constuct error quaternion
  qe = qhat.conjugate()*qin;

  double sgn = 1.;
  if (qe.w() < 0) {
    sgn = -1;
  }

  // Construct quaternion correction
  qcorr.w() = 1 - abs(qe.w());
  qcorr.vec() = sgn*qe.vec();
  qcorr = qhat * qcorr;

  Eigen::Vector3f err = pin - this->state.p;
  Eigen::Vector3f err_body;

  err_body = qhat.conjugate()._transformVector(err);

  double abias_max = this->geo_abias_max_;
  double gbias_max = this->geo_gbias_max_;

  // Update accel bias
  this->state.b.accel -= dt * this->geo_Kab_ * err_body;
  this->state.b.accel = this->state.b.accel.array().min(abias_max).max(-abias_max);

  // Update gyro bias
  this->state.b.gyro[0] -= dt * this->geo_Kgb_ * qe.w() * qe.x();
  this->state.b.gyro[1] -= dt * this->geo_Kgb_ * qe.w() * qe.y();
  this->state.b.gyro[2] -= dt * this->geo_Kgb_ * qe.w() * qe.z();
  this->state.b.gyro = this->state.b.gyro.array().min(gbias_max).max(-gbias_max);

  // Update state
  this->state.p += dt * this->geo_Kp_ * err;
  this->state.v.lin.w += dt * this->geo_Kv_ * err;

  this->state.q.w() += dt * this->geo_Kq_ * qcorr.w();
  this->state.q.x() += dt * this->geo_Kq_ * qcorr.x();
  this->state.q.y() += dt * this->geo_Kq_ * qcorr.y();
  this->state.q.z() += dt * this->geo_Kq_ * qcorr.z();
  this->state.q.normalize();

  // store previous pose, orientation, and velocity
  this->geo.prev_p = this->state.p;
  this->geo.prev_q = this->state.q;
  this->geo.prev_vel = this->state.v.lin.w;

}

sensor_msgs::Imu::Ptr dlio::LocNode::transformImu(const sensor_msgs::Imu::ConstPtr& imu_raw) {

  sensor_msgs::Imu::Ptr imu (new sensor_msgs::Imu);

  // Copy header
  imu->header = imu_raw->header;

  static double prev_stamp = imu->header.stamp.toSec();
  double dt = imu->header.stamp.toSec() - prev_stamp;
  prev_stamp = imu->header.stamp.toSec();

  if (dt == 0) { dt = 1.0/200.0; }

  // Transform angular velocity (will be the same on a rigid body, so just rotate to ROS convention)
  Eigen::Vector3f ang_vel(imu_raw->angular_velocity.x,
                          imu_raw->angular_velocity.y,
                          imu_raw->angular_velocity.z);

  Eigen::Vector3f ang_vel_cg = this->extrinsics.baselink2imu.R * ang_vel;

  imu->angular_velocity.x = ang_vel_cg[0];
  imu->angular_velocity.y = ang_vel_cg[1];
  imu->angular_velocity.z = ang_vel_cg[2];

  static Eigen::Vector3f ang_vel_cg_prev = ang_vel_cg;

  // Transform linear acceleration (need to account for component due to translational difference)
  Eigen::Vector3f lin_accel(imu_raw->linear_acceleration.x,
                            imu_raw->linear_acceleration.y,
                            imu_raw->linear_acceleration.z);

  Eigen::Vector3f lin_accel_cg = this->extrinsics.baselink2imu.R * lin_accel;

  lin_accel_cg = lin_accel_cg
                 + ((ang_vel_cg - ang_vel_cg_prev) / dt).cross(-this->extrinsics.baselink2imu.t)
                 + ang_vel_cg.cross(ang_vel_cg.cross(-this->extrinsics.baselink2imu.t));

  ang_vel_cg_prev = ang_vel_cg;

  imu->linear_acceleration.x = lin_accel_cg[0];
  imu->linear_acceleration.y = lin_accel_cg[1];
  imu->linear_acceleration.z = lin_accel_cg[2];

  return imu;

}

void dlio::LocNode::debug() {

  // Total length traversed
  double length_traversed = 0.;
  Eigen::Vector3f p_curr = Eigen::Vector3f(0., 0., 0.);
  Eigen::Vector3f p_prev = Eigen::Vector3f(0., 0., 0.);
  for (const auto& t : this->trajectory) {
    if (p_prev == Eigen::Vector3f(0., 0., 0.)) {
      p_prev = t.first;
      continue;
    }
    p_curr = t.first;
    double l = sqrt(pow(p_curr[0] - p_prev[0], 2) + pow(p_curr[1] - p_prev[1], 2) + pow(p_curr[2] - p_prev[2], 2));

    if (l >= 0.1) {
      length_traversed += l;
      p_prev = p_curr;
    }
  }
  this->length_traversed = length_traversed;

  // Average computation time
  double avg_comp_time =
    std::accumulate(this->comp_times.begin(), this->comp_times.end(), 0.0) / this->comp_times.size();

  // Average sensor rates
  int win_size = 100;
  double avg_imu_rate;
  double avg_lidar_rate;
  if (this->imu_rates.size() < win_size) {
    avg_imu_rate =
      std::accumulate(this->imu_rates.begin(), this->imu_rates.end(), 0.0) / this->imu_rates.size();
  } else {
    avg_imu_rate =
      std::accumulate(this->imu_rates.end()-win_size, this->imu_rates.end(), 0.0) / win_size;
  }
  if (this->lidar_rates.size() < win_size) {
    avg_lidar_rate =
      std::accumulate(this->lidar_rates.begin(), this->lidar_rates.end(), 0.0) / this->lidar_rates.size();
  } else {
    avg_lidar_rate =
      std::accumulate(this->lidar_rates.end()-win_size, this->lidar_rates.end(), 0.0) / win_size;
  }

  // RAM Usage
  double vm_usage = 0.0;
  double resident_set = 0.0;
  std::ifstream stat_stream("/proc/self/stat", std::ios_base::in); //get info from proc directory
  std::string pid, comm, state, ppid, pgrp, session, tty_nr;
  std::string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  std::string utime, stime, cutime, cstime, priority, nice;
  std::string num_threads, itrealvalue, starttime;
  unsigned long vsize;
  long rss;
  stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
              >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
              >> utime >> stime >> cutime >> cstime >> priority >> nice
              >> num_threads >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest
  stat_stream.close();
  long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // for x86-64 is configured to use 2MB pages
  vm_usage = vsize / 1024.0;
  resident_set = rss * page_size_kb;

  // CPU Usage
  struct tms timeSample;
  clock_t now;
  double cpu_percent;
  now = times(&timeSample);
  if (now <= this->lastCPU || timeSample.tms_stime < this->lastSysCPU ||
      timeSample.tms_utime < this->lastUserCPU) {
      cpu_percent = -1.0;
  } else {
      cpu_percent = (timeSample.tms_stime - this->lastSysCPU) + (timeSample.tms_utime - this->lastUserCPU);
      cpu_percent /= (now - this->lastCPU);
      cpu_percent /= this->numProcessors;
      cpu_percent *= 100.;
  }
  this->lastCPU = now;
  this->lastSysCPU = timeSample.tms_stime;
  this->lastUserCPU = timeSample.tms_utime;
  this->cpu_percents.push_back(cpu_percent);
  double avg_cpu_usage =
    std::accumulate(this->cpu_percents.begin(), this->cpu_percents.end(), 0.0) / this->cpu_percents.size();

  // Print to terminal
  printf("\033[2J\033[1;1H");

  std::cout << std::endl
            << "+-------------------------------------------------------------------+" << std::endl;
  std::cout << "|          Direct LiDAR-Inertial Localization v" << this->version_  << "              |"
            << std::endl;
  std::cout << "+-------------------------------------------------------------------+" << std::endl;

  std::time_t curr_time = this->scan_stamp;
  std::string asc_time = std::asctime(std::localtime(&curr_time)); asc_time.pop_back();
  std::cout << "| " << std::left << asc_time;
  std::cout << std::right << std::setfill(' ') << std::setw(42)
    << "Elapsed Time: " + to_string_with_precision(this->elapsed_time, 2) + " seconds "
    << "|" << std::endl;

  if ( !this->cpu_type.empty() ) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << this->cpu_type + " x " + std::to_string(this->numProcessors)
      << "|" << std::endl;
  }

  if (this->sensor == dlio::SensorType::OUSTER) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Ouster @ " + to_string_with_precision(avg_lidar_rate, 2)
                                   + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else if (this->sensor == dlio::SensorType::VELODYNE) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Velodyne @ " + to_string_with_precision(avg_lidar_rate, 2)
                                     + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else if (this->sensor == dlio::SensorType::HESAI) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Hesai @ " + to_string_with_precision(avg_lidar_rate, 2)
                                  + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else if (this->sensor == dlio::SensorType::LIVOX) {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Livox @ " + to_string_with_precision(avg_lidar_rate, 2)
                                  + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  } else {
    std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
      << "Sensor Rates: Unknown LiDAR @ " + to_string_with_precision(avg_lidar_rate, 2)
                                          + " Hz, IMU @ " + to_string_with_precision(avg_imu_rate, 2) + " Hz"
      << "|" << std::endl;
  }

  std::cout << "|===================================================================|" << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Position     {W}  [xyz] :: " + to_string_with_precision(this->state.p[0], 4) + " "
                                + to_string_with_precision(this->state.p[1], 4) + " "
                                + to_string_with_precision(this->state.p[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Orientation  {W} [wxyz] :: " + to_string_with_precision(this->state.q.w(), 4) + " "
                                + to_string_with_precision(this->state.q.x(), 4) + " "
                                + to_string_with_precision(this->state.q.y(), 4) + " "
                                + to_string_with_precision(this->state.q.z(), 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Lin Velocity {B}  [xyz] :: " + to_string_with_precision(this->state.v.lin.b[0], 4) + " "
                                + to_string_with_precision(this->state.v.lin.b[1], 4) + " "
                                + to_string_with_precision(this->state.v.lin.b[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Ang Velocity {B}  [xyz] :: " + to_string_with_precision(this->state.v.ang.b[0], 4) + " "
                                + to_string_with_precision(this->state.v.ang.b[1], 4) + " "
                                + to_string_with_precision(this->state.v.ang.b[2], 4)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Accel Bias        [xyz] :: " + to_string_with_precision(this->state.b.accel[0], 8) + " "
                                + to_string_with_precision(this->state.b.accel[1], 8) + " "
                                + to_string_with_precision(this->state.b.accel[2], 8)
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Gyro Bias         [xyz] :: " + to_string_with_precision(this->state.b.gyro[0], 8) + " "
                                + to_string_with_precision(this->state.b.gyro[1], 8) + " "
                                + to_string_with_precision(this->state.b.gyro[2], 8)
    << "|" << std::endl;

  std::cout << "|                                                                   |" << std::endl;

  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Distance Traveled  :: " + to_string_with_precision(length_traversed, 4) + " meters"
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Distance to Origin :: "
      + to_string_with_precision( sqrt(pow(this->state.p[0]-this->origin[0],2) +
                                       pow(this->state.p[1]-this->origin[1],2) +
                                       pow(this->state.p[2]-this->origin[2],2)), 4) + " meters"
    << "|" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "Registration       :: global map points: " + std::to_string(this->globalmap ? this->globalmap->points.size() : 0)
    << "|" << std::endl;
  std::cout << "|                                                                   |" << std::endl;

  double avg_gicp_time = 0.;
  if (!this->gicp_times.empty()) {
    avg_gicp_time = std::accumulate(this->gicp_times.begin(), this->gicp_times.end(), 0.0) / this->gicp_times.size();
  }

  std::cout << std::right << std::setprecision(2) << std::fixed;
  std::cout << "| Computation Time :: "
    << std::setfill(' ') << std::setw(6) << this->comp_times.back()*1000. << " ms    // Avg: "
    << std::setw(6) << avg_comp_time*1000. << " / Max: "
    << std::setw(6) << *std::max_element(this->comp_times.begin(), this->comp_times.end())*1000.
    << "     |" << std::endl;
  if (!this->gicp_times.empty()) {
  std::cout << "| GICP Align Time  :: "
    << std::setfill(' ') << std::setw(6) << this->gicp_times.back()*1000. << " ms    // Avg: "
    << std::setw(6) << avg_gicp_time*1000. << " / Max: "
    << std::setw(6) << *std::max_element(this->gicp_times.begin(), this->gicp_times.end())*1000.
    << "     |" << std::endl;
  }
  std::cout << "| Cores Utilized   :: "
    << std::setfill(' ') << std::setw(6) << (cpu_percent/100.) * this->numProcessors << " cores // Avg: "
    << std::setw(6) << (avg_cpu_usage/100.) * this->numProcessors << " / Max: "
    << std::setw(6) << (*std::max_element(this->cpu_percents.begin(), this->cpu_percents.end()) / 100.)
                       * this->numProcessors
    << "     |" << std::endl;
  std::cout << "| CPU Load         :: "
    << std::setfill(' ') << std::setw(6) << cpu_percent << " %     // Avg: "
    << std::setw(6) << avg_cpu_usage << " / Max: "
    << std::setw(6) << *std::max_element(this->cpu_percents.begin(), this->cpu_percents.end())
    << "     |" << std::endl;
  std::cout << "| " << std::left << std::setfill(' ') << std::setw(66)
    << "RAM Allocation   :: " + to_string_with_precision(resident_set/1000., 2) + " MB"
    << "|" << std::endl;

  std::cout << "+-------------------------------------------------------------------+" << std::endl;

}
