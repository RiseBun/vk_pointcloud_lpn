#ifndef _GRID_MAPPER_HPP
#define _GRID_MAPPER_HPP

#include <vk_sdk/Sdk.hpp>
#include <vk_sdk/capnp/pointcloud.capnp.h>
#include <vk_sdk/capnp/header.capnp.h>
#include <vk_sdk/Utilities.hpp>
#include <vk_sdk/Receivers.hpp>
#include <vk_sdk/VisualKit.hpp>

#include <vk_sdk/DisparityToPointCloud.hpp>

#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <iostream>
#include <random>
#include <queue>
#include <tuple>
#include <memory>

#include "raycast.hpp"

#define logit(x) (log((x) / (1 - (x))))

using namespace std;

// voxel hashing
template <typename T>
struct matrix_hash : std::unary_function<T, size_t>
{
  std::size_t operator()(T const &matrix) const
  {
    size_t seed = 0;
    for (size_t i = 0; i < matrix.size(); ++i)
    {
      auto elem = *(matrix.data() + i);
      seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

// constant parameters
struct GridMapperParams
{
  std::string tf_prefix = "";
  double resolution_ = -1.0;
  double map_size_x_ = -1.0;
  double map_size_y_ = -1.0;
  double map_size_z_ = -1.0;
  double local_update_range_x_ = -1.0;
  double local_update_range_y_ = -1.0;
  double local_update_range_z_ = -1.0;
  double obstacles_inflation_ = -1.0;
  double p_hit_ = 0.70;
  double p_miss_ = 0.35;
  double p_min_ = 0.12;
  double p_max_ = 0.97;
  double p_occ_ = 0.8;
  double min_ray_length_ = -0.1;
  double max_ray_length_ = -0.1;
  double local_map_margin_ = 1;
  std::string device_version_ = "180";
  std::string disparity_suffix_ = "disparity";
  std::string input_cloud_topic_ = "";
  std::string output_topic_ = "pointcloud";
  double odom_depth_timeout_ = 1.0;
  double visualization_truncate_height_ = 2.0;
  double fading_time_ = 1000.0;
  int skip_pixel_ = 1;
};

struct MappingParameters
{

  /* map properties */
  Eigen::Vector3d map_origin_, map_size_;
  bool flip_x_, flip_y_, flip_z_; // if true, axis starts in middle and circles back to middle
  Eigen::Vector3d map_min_boundary_, map_max_boundary_; // map range in pos
  Eigen::Vector3d map_remain_min_boundary_, map_remain_max_boundary_; // map range in pos
  Eigen::Vector3i map_voxel_num_;                       // map range in index
  Eigen::Vector3d local_update_range_;
  double resolution_, resolution_inv_;
  double obstacles_inflation_;
  string frame_id_;
  int pose_type_;

  /* camera parameters */
  double cx_, cy_, fx_, fy_;

  /* time out */
  double odom_depth_timeout_;

  /* depth image projection filtering */
  // double depth_filter_mindist_, depth_filter_tolerance_;
  // int depth_filter_margin_;
  // bool use_depth_filter_;
  // double k_depth_scaling_factor_;
  int skip_pixel_;

  /* raycasting */
  double p_hit_, p_miss_, p_min_, p_max_, p_occ_; // occupancy probability
  double prob_hit_log_, prob_miss_log_, clamp_min_log_, clamp_max_log_,
      min_occupancy_log_;                  // logit of occupancy probability
  double min_ray_length_, max_ray_length_; // range of doing raycasting

  /* local map update and clear */
  int local_map_margin_;

  /* visualization and computation time display */
  double visualization_truncate_height_, virtual_ceil_height_, ground_height_, virtual_ceil_yp_, virtual_ceil_yn_;
  bool show_occ_time_;
  double fading_time_;

  /* active mapping */
  double unknown_flag_;
};

// intermediate mapping data for fusion

struct MappingData
{
  // main map data, occupancy of each voxel and Euclidean distance

  std::vector<double> occupancy_buffer_;
  std::vector<char> occupancy_buffer_inflate_;

  // camera position and pose data

  Eigen::Vector3d camera_pos_, last_camera_pos_;
  Eigen::Matrix3d camera_r_m_, last_camera_r_m_;
  Eigen::Matrix4d cam2body_;

  // depth image data

  //   cv::Mat depth_image_, last_depth_image_;
  int image_cnt_;

  // flags of map state

  bool occ_need_update_, local_updated_;
  bool has_first_depth_;
  bool has_odom_, has_cloud_;

  // odom_depth_timeout_
  //   ros::Time last_occ_update_time_;
  uint64_t latest_stamp_;
  uint64_t last_occ_update_time_;
  vkc::Shared<vkc::Header> latest_header_;
  bool flag_depth_odom_timeout_;
  bool flag_use_depth_fusion;

  // depth image projected point cloud

  vector<Eigen::Vector3d> proj_points_;
  int proj_points_cnt;

  // flag buffers for speeding up raycasting

  vector<short> count_hit_, count_hit_and_miss_;
  vector<char> flag_traverse_, flag_rayend_;
  char raycast_num_;
  queue<Eigen::Vector3i> cache_voxel_;

  // range of updating grid

  Eigen::Vector3i local_bound_min_, local_bound_max_;

  // computation time

  double fuse_time_, max_fuse_time_;
  int update_num_;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class PointCloudGenerator;


class GridMap
{
public:
  GridMap() {}
  ~GridMap() {}

  friend class PointCloudGenerator;

  enum
  {
    POSE_STAMPED = 1,
    ODOMETRY = 2,
    INVALID_IDX = -10000
  };

  // occupancy map management
  void resetBuffer();
  void resetBuffer(Eigen::Vector3d min, Eigen::Vector3d max);

  inline uint64_t getLatestStamp() { return md_.latest_stamp_; }
  inline double toSecs(uint64_t stamp) { return stamp / 1e9; }
  inline vkc::Shared<vkc::Header> getLatestHeader() {
    return md_.latest_header_;
  }
  inline void setLatestHeader(vkc::Shared<vkc::Header> header) {
    md_.latest_header_ = header;
  }
  void setCameraPosition(const Eigen::Vector3d &pos);
  inline void updateCam2Body(const Eigen::Matrix4d &cam2body) {
    md_.cam2body_ = cam2body;
  };

  inline void posToIndex(const Eigen::Vector3d &pos, Eigen::Vector3i &id);
  inline void indexToPos(const Eigen::Vector3i &id, Eigen::Vector3d &pos);
  inline int toAddress(const Eigen::Vector3i &id);
  inline int toAddress(int &x, int &y, int &z);
  inline bool isInMap(const Eigen::Vector3d &pos);
  inline bool isInMap(const Eigen::Vector3i &idx);

  inline void setOccupancy(Eigen::Vector3d pos, double occ = 1);
  inline void setOccupied(Eigen::Vector3d pos);
  inline int getOccupancy(Eigen::Vector3d pos);
  inline int getOccupancy(Eigen::Vector3i id);
  inline int getInflateOccupancy(Eigen::Vector3d pos);
  inline bool shouldUpdateMap(const Eigen::Vector3d &pos);
  inline void updateMap(const Eigen::Vector3d &pos);

  inline auto getOffset() {
    return Eigen::Vector3d(mp_.map_size_.x() / 2, mp_.map_size_.y() / 2, mp_.map_size_.z() / 2);
  }

  inline void boundIndex(Eigen::Vector3i &id);
  inline bool isUnknown(const Eigen::Vector3i &id);
  inline bool isUnknown(const Eigen::Vector3d &pos);
  inline bool isKnownFree(const Eigen::Vector3i &id);
  inline bool isKnownOccupied(const Eigen::Vector3i &id);

  void initMap(GridMapperParams &cfg);

  void publishMap();
  void publishMapInflate(bool all_info = false);
  void updateOccupancy();

  vkc::Shared<vkc::PointCloud> convertToCapnpCloud(
      std::shared_ptr<open3d::geometry::PointCloud> cloud,
      vkc::Shared<vkc::Header> header);

  void publishDepth();

  bool hasDepthObservation();
  bool odomValid();
  void getRegion(Eigen::Vector3d &ori, Eigen::Vector3d &size);
  inline double getResolution();
  Eigen::Vector3d getOrigin();
  int getVoxelNum();
  inline bool getOdomDepthTimeout() { return md_.flag_depth_odom_timeout_; }

  std::reference_wrapper<vkc::DataSource> getVkcSource();
  void runVkc();

  void fade();

  using Ptr = std::shared_ptr<GridMap>;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  MappingParameters mp_;
  MappingData md_;
  std::optional<std::reference_wrapper<vkc::DataSource>> vkc_source_ref_;

  std::string output_topic_;
  std::unique_ptr<vkc::VisualKit> visualkit;
  std::shared_ptr<vkc::Receiver<vkc::PointCloud>> map_receiver;
  std::shared_ptr<vkc::Receiver<vkc::PointCloud>> map_inflate_receiver;

  // get depth image and camera pose
  //   void depthPoseCallback(const sensor_msgs::ImageConstPtr& img,
  //                          const geometry_msgs::PoseStampedConstPtr& pose);
  //   void extrinsicCallback(const nav_msgs::OdometryConstPtr& odom);
  //   void depthOdomCallback(const sensor_msgs::ImageConstPtr& img, const nav_msgs::OdometryConstPtr& odom);
  //   void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& img);
  //   void odomCallback(const nav_msgs::OdometryConstPtr& odom);

  // update occupancy by raycasting
  //   void visCallback(const ros::TimerEvent& /*event*/);

  // main update process
  // void projectDepthImage();
  // set md_.proj_points, md_.proj_points_cnt
  void raycastProcess();
  void clearAndInflateLocalMap();

  inline void inflatePoint(const Eigen::Vector3i &pt, int step, vector<Eigen::Vector3i> &pts);
  int setCacheOccupancy(Eigen::Vector3d pos, int occ);
  Eigen::Vector3d closestPointInMap(const Eigen::Vector3d &pt, const Eigen::Vector3d &camera_pt);

  // typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image,
  // nav_msgs::Odometry> SyncPolicyImageOdom; typedef
  // message_filters::sync_policies::ExactTime<sensor_msgs::Image,
  // geometry_msgs::PoseStamped> SyncPolicyImagePose;
  //   typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry>
  //       SyncPolicyImageOdom;
  //   typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, geometry_msgs::PoseStamped>
  //       SyncPolicyImagePose;
  //   typedef shared_ptr<message_filters::Synchronizer<SyncPolicyImagePose>> SynchronizerImagePose;
  //   typedef shared_ptr<message_filters::Synchronizer<SyncPolicyImageOdom>> SynchronizerImageOdom;

  //   ros::NodeHandle node_;
  //   shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> depth_sub_;
  //   shared_ptr<message_filters::Subscriber<geometry_msgs::PoseStamped>> pose_sub_;
  //   shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_sub_;
  //   SynchronizerImagePose sync_image_pose_;
  //   SynchronizerImageOdom sync_image_odom_;

  //   ros::Subscriber indep_cloud_sub_, indep_odom_sub_, extrinsic_sub_;
  //   ros::Publisher map_pub_, map_inf_pub_;
  //   ros::Timer occ_timer_, vis_timer_;

  //
  uniform_real_distribution<double> rand_noise_;
  normal_distribution<double> rand_noise2_;
  default_random_engine eng_;

  std::mutex update_map_mutex_;
};

/* ============================== definition of inline function
 * ============================== */

inline int GridMap::toAddress(const Eigen::Vector3i &id)
{
  int x = id(0), y = id(1), z = id(2);
  if (mp_.flip_x_)
    x = (x + mp_.map_voxel_num_(0) / 2) % mp_.map_voxel_num_(0);
  if (mp_.flip_y_)
    y = (y + mp_.map_voxel_num_(1) / 2) % mp_.map_voxel_num_(1);
  if (mp_.flip_z_)
    z = (z + mp_.map_voxel_num_(2) / 2) % mp_.map_voxel_num_(2);
  return x * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2) + y * mp_.map_voxel_num_(2) + z;
}

inline int GridMap::toAddress(int &x, int &y, int &z)
{
  int x1 = x, y1 = y, z1 = z;
  if (mp_.flip_x_)
    x1 = (x1 + mp_.map_voxel_num_(0) / 2) % mp_.map_voxel_num_(0);
  if (mp_.flip_y_)
    y1 = (y1 + mp_.map_voxel_num_(1) / 2) % mp_.map_voxel_num_(1);
  if (mp_.flip_z_)
    z1 = (z1 + mp_.map_voxel_num_(2) / 2) % mp_.map_voxel_num_(2);
  return x1 * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2) + y1 * mp_.map_voxel_num_(2) + z1;
}

inline void GridMap::boundIndex(Eigen::Vector3i &id)
{
  Eigen::Vector3i id1;
  id1(0) = max(min(id(0), mp_.map_voxel_num_(0) - 1), 0);
  id1(1) = max(min(id(1), mp_.map_voxel_num_(1) - 1), 0);
  id1(2) = max(min(id(2), mp_.map_voxel_num_(2) - 1), 0);
  id = id1;
}

inline bool GridMap::isUnknown(const Eigen::Vector3i &id)
{
  Eigen::Vector3i id1 = id;
  boundIndex(id1);
  return md_.occupancy_buffer_[toAddress(id1)] < mp_.clamp_min_log_ - 1e-3;
}

inline bool GridMap::isUnknown(const Eigen::Vector3d &pos)
{
  Eigen::Vector3i idc;
  posToIndex(pos, idc);
  return isUnknown(idc);
}

inline bool GridMap::isKnownFree(const Eigen::Vector3i &id)
{
  Eigen::Vector3i id1 = id;
  boundIndex(id1);
  int adr = toAddress(id1);

  // return md_.occupancy_buffer_[adr] >= mp_.clamp_min_log_ &&
  //     md_.occupancy_buffer_[adr] < mp_.min_occupancy_log_;
  return md_.occupancy_buffer_[adr] >= mp_.clamp_min_log_ && md_.occupancy_buffer_inflate_[adr] == 0;
}

inline bool GridMap::isKnownOccupied(const Eigen::Vector3i &id)
{
  Eigen::Vector3i id1 = id;
  boundIndex(id1);
  int adr = toAddress(id1);

  return md_.occupancy_buffer_inflate_[adr] == 1;
}

inline void GridMap::setOccupied(Eigen::Vector3d pos)
{
  if (!isInMap(pos))
    return;

  Eigen::Vector3i id;
  posToIndex(pos, id);

  md_.occupancy_buffer_inflate_[id(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2) +
                                id(1) * mp_.map_voxel_num_(2) + id(2)] = 1;
}

inline void GridMap::setOccupancy(Eigen::Vector3d pos, double occ)
{
  if (occ != 1 && occ != 0)
  {
    cout << "occ value error!" << endl;
    return;
  }

  if (!isInMap(pos))
    return;

  Eigen::Vector3i id;
  posToIndex(pos, id);

  md_.occupancy_buffer_[toAddress(id)] = occ;
}

inline int GridMap::getOccupancy(Eigen::Vector3d pos)
{
  if (!isInMap(pos))
    return -1;

  Eigen::Vector3i id;
  posToIndex(pos, id);

  return md_.occupancy_buffer_[toAddress(id)] > mp_.min_occupancy_log_ ? 1 : 0;
}

inline int GridMap::getInflateOccupancy(Eigen::Vector3d pos)
{
  if (!isInMap(pos))
    return -1;

  Eigen::Vector3i id;
  posToIndex(pos, id);

  return int(md_.occupancy_buffer_inflate_[toAddress(id)]);
}

inline int GridMap::getOccupancy(Eigen::Vector3i id)
{
  if (id(0) < 0 || id(0) >= mp_.map_voxel_num_(0) || id(1) < 0 || id(1) >= mp_.map_voxel_num_(1) ||
      id(2) < 0 || id(2) >= mp_.map_voxel_num_(2))
    return -1;

  return md_.occupancy_buffer_[toAddress(id)] > mp_.min_occupancy_log_ ? 1 : 0;
}

inline bool GridMap::shouldUpdateMap(const Eigen::Vector3d &pos)
{
  if (pos(0) < mp_.map_remain_min_boundary_(0) + 1e-4 || pos(1) < mp_.map_remain_min_boundary_(1) + 1e-4 ||
      pos(2) < mp_.map_remain_min_boundary_(2) + 1e-4)
  {
    return true;
  }
  if (pos(0) > mp_.map_remain_max_boundary_(0) - 1e-4 || pos(1) > mp_.map_remain_max_boundary_(1) - 1e-4 ||
      pos(2) > mp_.map_remain_max_boundary_(2) - 1e-4)
  {
    return true;
  }
  return false;
}

inline void GridMap::updateMap(const Eigen::Vector3d &pos)
{
  bool keep_x0 {true}, keep_x1 {true},
       keep_y0 {true}, keep_y1 {true},
       keep_z0 {true}, keep_z1 {true};


  if (pos(0) < mp_.map_remain_min_boundary_(0) + 1e-4)
    keep_x1 = false;
  if (pos(0) > mp_.map_remain_max_boundary_(0) - 1e-4)
    keep_x0 = false;
  if (pos(1) < mp_.map_remain_min_boundary_(1) + 1e-4)
    keep_y1 = false;
  if (pos(1) > mp_.map_remain_max_boundary_(1) - 1e-4)
    keep_y0 = false;
  if (pos(2) < mp_.map_remain_min_boundary_(2) + 1e-4)
    keep_z1 = false;
  if (pos(2) > mp_.map_remain_max_boundary_(2) - 1e-4)
    keep_z0 = false;
  
  if (!isInMap(pos)) {
    keep_x0 = false;
    keep_x1 = false;
    keep_y0 = false;
    keep_y1 = false;
    keep_z0 = false;
    keep_z1 = false;

    int buffer_size = mp_.map_voxel_num_(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2);
    md_.occupancy_buffer_ = vector<double>(buffer_size, mp_.clamp_min_log_ - mp_.unknown_flag_);
    md_.occupancy_buffer_inflate_ = vector<char>(buffer_size, 0);

    // teleport map boundaries
    int x_i = floor((pos(0) - mp_.map_size_(0) / 2) / (mp_.map_size_(0) / 2));
    mp_.map_origin_(0) = x_i * mp_.map_size_(0) / 2;
    int y_i = floor((pos(1) - mp_.map_size_(1) / 2) / (mp_.map_size_(1) / 2));
    mp_.map_origin_(1) = y_i * mp_.map_size_(1) / 2;
    int z_i = floor((pos(2) - mp_.map_size_(2) / 2) / (mp_.map_size_(2) / 2));
    mp_.map_origin_(2) = z_i * mp_.map_size_(2) / 2;
    mp_.map_min_boundary_ = mp_.map_origin_;
    mp_.map_max_boundary_ = mp_.map_origin_ + mp_.map_size_;
    mp_.map_remain_min_boundary_ = mp_.map_origin_ + mp_.map_size_ / 8;
    mp_.map_remain_max_boundary_ = mp_.map_origin_ + mp_.map_size_ * 7 / 8;
    mp_.flip_x_ = x_i % 2 == 1;
    mp_.flip_y_ = y_i % 2 == 1;
    mp_.flip_z_ = z_i % 2 == 1;
    return;
  }

  // clear unwanted points
  // spdlog::info("clearing points {} {} {} {} {} {}", keep_x0, keep_x1, keep_y0, keep_y1, keep_z0, keep_z1);
  // spdlog::info("pos: {} {} {}", pos(0), pos(1), pos(2));
  // spdlog::info("old boundary and remain boundary {} {} {} {}",
  //              mp_.map_min_boundary_(0), mp_.map_max_boundary_(0),
  //              mp_.map_remain_min_boundary_(0), mp_.map_remain_max_boundary_(0));
  // spdlog::info("old boundary and remain boundary {} {} {} {}",
  //              mp_.map_min_boundary_(1), mp_.map_max_boundary_(1),
  //              mp_.map_remain_min_boundary_(1), mp_.map_remain_max_boundary_(1));
  // spdlog::info("old boundary and remain boundary {} {} {} {}",
  //              mp_.map_min_boundary_(2), mp_.map_max_boundary_(2),
  //              mp_.map_remain_min_boundary_(2), mp_.map_remain_max_boundary_(2));
  for (int x = 0; x < mp_.map_voxel_num_(0); ++x)
    for (int y = 0; y < mp_.map_voxel_num_(1); ++y)
      for (int z = 0; z < mp_.map_voxel_num_(2); ++z)
      {
        if (x < mp_.map_voxel_num_(0)/2 + 1 && !keep_x0)
        {
          int idx = toAddress(x, y, z);
          md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
          md_.occupancy_buffer_inflate_[idx] = 0;
        }
        if (x >=mp_.map_voxel_num_(0)/2 - 1 && !keep_x1)
        {
          int idx = toAddress(x, y, z);
          md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
          md_.occupancy_buffer_inflate_[idx] = 0;
        }
        if (y < mp_.map_voxel_num_(1)/2 + 1 && !keep_y0)
        {
          int idx = toAddress(x, y, z);
          md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
          md_.occupancy_buffer_inflate_[idx] = 0;
        }
        if (y >= mp_.map_voxel_num_(1)/2 - 1 && !keep_y1)
        {
          int idx = toAddress(x, y, z);
          md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
          md_.occupancy_buffer_inflate_[idx] = 0;
        }
        if (z < mp_.map_voxel_num_(2)/2 + 1 && !keep_z0)
        {
          int idx = toAddress(x, y, z);
          md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
          md_.occupancy_buffer_inflate_[idx] = 0;
        }
        if (z >= mp_.map_voxel_num_(2)/2 + 1 && !keep_z1)
        {
          int idx = toAddress(x, y, z);
          md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
          md_.occupancy_buffer_inflate_[idx] = 0;
        }
      }

  if (pos(0) < mp_.map_remain_min_boundary_(0) + 1e-4)
  {
    double diff = mp_.map_size_(0) / 2;
    mp_.map_min_boundary_(0) -= diff;
    mp_.map_max_boundary_(0) -= diff;
    mp_.map_remain_min_boundary_(0) -= diff;
    mp_.map_remain_max_boundary_(0) -= diff;
    mp_.map_origin_(0) -= diff;
    mp_.flip_x_ = !mp_.flip_x_;
  }
  if (pos(0) > mp_.map_remain_max_boundary_(0) - 1e-4)
  {
    double diff = mp_.map_size_(0) / 2;
    mp_.map_min_boundary_(0) += diff;
    mp_.map_max_boundary_(0) += diff;
    mp_.map_remain_min_boundary_(0) += diff;
    mp_.map_remain_max_boundary_(0) += diff;
    mp_.map_origin_(0) += diff;
    mp_.flip_x_ = !mp_.flip_x_;
  }
  if (pos(1) < mp_.map_remain_min_boundary_(1) + 1e-4)
  {
    double diff = mp_.map_size_(1) / 2;
    mp_.map_min_boundary_(1) -= diff;
    mp_.map_max_boundary_(1) -= diff;
    mp_.map_remain_min_boundary_(1) -= diff;
    mp_.map_remain_max_boundary_(1) -= diff;
    mp_.map_origin_(1) -= diff;
    mp_.flip_y_ = !mp_.flip_y_;
  }
  if (pos(1) > mp_.map_remain_max_boundary_(1) - 1e-4)
  {
    double diff = mp_.map_size_(1) / 2;
    mp_.map_min_boundary_(1) += diff;
    mp_.map_max_boundary_(1) += diff;
    mp_.map_remain_min_boundary_(1) += diff;
    mp_.map_remain_max_boundary_(1) += diff;
    mp_.map_origin_(1) += diff;
    mp_.flip_y_ = !mp_.flip_y_;
  }
  if (pos(2) < mp_.map_remain_min_boundary_(2) + 1e-4)
  {
    double diff = mp_.map_size_(2) / 2;
    mp_.map_min_boundary_(2) -= diff;
    mp_.map_max_boundary_(2) -= diff;
    mp_.map_remain_min_boundary_(2) -= diff;
    mp_.map_remain_max_boundary_(2) -= diff;
    mp_.map_origin_(2) -= diff;
    mp_.flip_z_ = !mp_.flip_z_;
  }
  if (pos(2) > mp_.map_remain_max_boundary_(2) - 1e-4)
  {
    double diff = mp_.map_size_(2) / 2;
    mp_.map_min_boundary_(2) += diff;
    mp_.map_max_boundary_(2) += diff;
    mp_.map_remain_min_boundary_(2) += diff;
    mp_.map_remain_max_boundary_(2) += diff;
    mp_.map_origin_(2) += diff;
    mp_.flip_z_ = !mp_.flip_z_;
  }

  // spdlog::info("new boundary and remain boundary {} {} {} {}",
  //              mp_.map_min_boundary_(0), mp_.map_max_boundary_(0),
  //              mp_.map_remain_min_boundary_(0), mp_.map_remain_max_boundary_(0));
  // spdlog::info("new boundary and remain boundary {} {} {} {}",
  //              mp_.map_min_boundary_(1), mp_.map_max_boundary_(1),
  //              mp_.map_remain_min_boundary_(1), mp_.map_remain_max_boundary_(1));
  // spdlog::info("new boundary and remain boundary {} {} {} {}",
  //              mp_.map_min_boundary_(2), mp_.map_max_boundary_(2),
  //              mp_.map_remain_min_boundary_(2), mp_.map_remain_max_boundary_(2));               
}

inline bool GridMap::isInMap(const Eigen::Vector3d& pos) {
  if (pos(0) < mp_.map_min_boundary_(0) + 1e-4 || pos(1) < mp_.map_min_boundary_(1) + 1e-4 ||
      pos(2) < mp_.map_min_boundary_(2) + 1e-4) {
    // cout << "less than min range!" << endl;
    return false;
  }
  if (pos(0) > mp_.map_max_boundary_(0) - 1e-4 || pos(1) > mp_.map_max_boundary_(1) - 1e-4 ||
      pos(2) > mp_.map_max_boundary_(2) - 1e-4) {
    return false;
  }
  return true;
}

inline bool GridMap::isInMap(const Eigen::Vector3i &idx)
{
  if (idx(0) < 0 || idx(1) < 0 || idx(2) < 0)
  {
    spdlog::info("isInMap: NO less than min range! {} {} {}", idx(0), idx(1), idx(2));
    return false;
  }
  if (idx(0) > mp_.map_voxel_num_(0) - 1 || idx(1) > mp_.map_voxel_num_(1) - 1 ||
      idx(2) > mp_.map_voxel_num_(2) - 1)
  {
    spdlog::info("isInMap: NO more than max range! {} {} {}", idx(0), idx(1), idx(2));
    return false;
  }
  return true;
}

inline void GridMap::posToIndex(const Eigen::Vector3d &pos, Eigen::Vector3i &id)
{
  for (int i = 0; i < 3; ++i) {
    id(i) = floor((pos(i) - mp_.map_origin_(i)) * mp_.resolution_inv_);
  }
}

inline void GridMap::indexToPos(const Eigen::Vector3i &id, Eigen::Vector3d &pos)
{
  for (int i = 0; i < 3; ++i) {
    pos(i) = (id(i) + 0.5) * mp_.resolution_ + mp_.map_origin_(i);
  }

}

inline void GridMap::inflatePoint(const Eigen::Vector3i &pt, int step, vector<Eigen::Vector3i> &pts)
{
  int num = 0;

  /* ---------- all inflate ---------- */
  for (int x = -step; x <= step; ++x)
    for (int y = -step; y <= step; ++y)
      for (int z = -step; z <= step; ++z)
      {
        pts[num++] = Eigen::Vector3i(pt(0) + x, pt(1) + y, pt(2) + z);
      }
}

inline double GridMap::getResolution() { return mp_.resolution_; }

inline void GridMap::fade() {
  Eigen::Vector3d local_range_min = md_.camera_pos_ - mp_.local_update_range_;
  Eigen::Vector3d local_range_max = md_.camera_pos_ + mp_.local_update_range_;

  Eigen::Vector3i min_id, max_id;
  posToIndex(local_range_min, min_id);
  posToIndex(local_range_max, max_id);
  boundIndex(min_id);
  boundIndex(max_id);

  const double reduce = (mp_.clamp_max_log_ - mp_.min_occupancy_log_) / (mp_.fading_time_ * 2); // function called at 2Hz
  const double low_thres = mp_.clamp_min_log_ + reduce;
  update_map_mutex_.lock();
  for (int x = min_id(0); x <= max_id(0); ++x)
    for (int y = min_id(1); y <= max_id(1); ++y)
      for (int z = min_id(2); z <= max_id(2); ++z)
      {
        int address = toAddress(x, y, z);
        if (md_.occupancy_buffer_[address] > low_thres)
        {
          md_.occupancy_buffer_[address] -= reduce;
        }
      }
  update_map_mutex_.unlock();
}
#endif
