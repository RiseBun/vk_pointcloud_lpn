#pragma once

#include <deque>
#include <map>
#include <optional>
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Geometry>

// 引入 vk_sdk 类型 (为了引用 Message/Shared 指针)
#include <vk_sdk/Sdk.hpp> 
// 注意：如果不想在这里引入整个 vk_sdk，可以使用前置声明或模板，但引入最方便

namespace trajectory {

// 同步后的数据包
struct SyncedPacket {
    int64_t timestamp_ns; // 统一后的时间戳 (通常以 Left Image 为准)
    
    // 原始数据指针 (使用 vkc::Shared 避免拷贝)
    vkc::Shared<vkc::Disparity> left_disp;
    vkc::Shared<vkc::Disparity> right_disp;
    vkc::Shared<vkc::Odometry3d> odom; // 插值后的 odom 还是原始 odom? 建议先存原始最近邻
    
    // 也可以存插值好的 Pose，方便 PoseGraph 直接用
    std::optional<Eigen::Isometry3d> interpolated_odom_pose;
};

class TimeSync {
public:
    TimeSync(int64_t cam_time_offset_ns);

    // 数据输入接口
    void addLeftDisp(vkc::Shared<vkc::Disparity> msg);
    void addRightDisp(vkc::Shared<vkc::Disparity> msg);
    void addOdom(vkc::Shared<vkc::Odometry3d> msg);

    // 尝试提取一组已同步的数据
    // 返回 nullopt 如果当前没有匹配好的数据
    std::optional<SyncedPacket> tryGetNextPacket();

private:
    int64_t offset_ns_;
    double sync_tolerance_ns_ = 20e6; // 20ms 容差

    // 缓冲队列 (时间戳 -> 数据)
    // 使用 map 自动排序，方便找 lower_bound
    std::map<int64_t, vkc::Shared<vkc::Disparity>> queue_left_;
    std::map<int64_t, vkc::Shared<vkc::Disparity>> queue_right_;
    std::map<int64_t, vkc::Shared<vkc::Odometry3d>> queue_odom_;
    
    // 用于记录处理进度，避免重复处理
    int64_t last_processed_time_ = 0;
};

} // namespace trajectory