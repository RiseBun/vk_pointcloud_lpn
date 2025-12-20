#include "trajectory/TimeSync.hpp"
#include <iostream>
#include <cmath>

namespace trajectory {

TimeSync::TimeSync(int64_t cam_time_offset_ns) 
    : offset_ns_(cam_time_offset_ns) {}

void TimeSync::addLeftDisp(vkc::Shared<vkc::Disparity> msg) {
    auto header = msg.reader().getHeader();
    int64_t t = header.getStampMonotonic() + header.getClockOffset();
    queue_left_[t] = msg;
}

void TimeSync::addRightDisp(vkc::Shared<vkc::Disparity> msg) {
    auto header = msg.reader().getHeader();
    int64_t t = header.getStampMonotonic() + header.getClockOffset();
    queue_right_[t] = msg;
}

void TimeSync::addOdom(vkc::Shared<vkc::Odometry3d> msg) {
    auto header = msg.reader().getHeader();
    int64_t t = header.getStampMonotonic() + header.getClockOffset();
    queue_odom_[t] = msg;
}

// 辅助函数：在 map 中找最近邻
template <typename T>
typename std::map<int64_t, T>::iterator findClosest(
    std::map<int64_t, T>& map, int64_t target_time, int64_t tolerance) {
    
    auto it = map.lower_bound(target_time); // >= target
    
    // 检查 it 和 prev(it) 谁更近
    auto best_it = map.end();
    int64_t min_diff = tolerance + 1;

    if (it != map.end()) {
        int64_t diff = std::abs(it->first - target_time);
        if (diff <= tolerance) {
            min_diff = diff;
            best_it = it;
        }
    }

    if (it != map.begin()) {
        auto prev = std::prev(it);
        int64_t diff = std::abs(prev->first - target_time);
        if (diff < min_diff && diff <= tolerance) {
            best_it = prev;
        }
    }
    return best_it;
}

std::optional<SyncedPacket> TimeSync::tryGetNextPacket() {
    if (queue_left_.empty()) return std::nullopt;

    // 1. 遍历左目队列作为基准
    // 这里简单处理：每次取最早的一个左目帧尝试匹配
    auto it_left = queue_left_.begin(); 
    int64_t t_left = it_left->first;
    
    // 如果这个帧比上次处理的时间还早，直接扔掉（乱序保护）
    if (t_left <= last_processed_time_) {
        queue_left_.erase(it_left);
        return tryGetNextPacket(); // 递归试下一个
    }

    // 2. 尝试匹配右目 (时间戳应该几乎一致)
    auto it_right = findClosest(queue_right_, t_left, 5e6); // 右目容差小一点，5ms
    if (it_right == queue_right_.end()) {
        // 右目还没来？或者丢帧了？
        // 策略：如果左目太老了（比如堆积了 100 帧），就丢弃；否则等待
        if (queue_left_.size() > 50) {
            std::cout << "[TimeSync] Drop Left frame " << t_left << " (No Right match)" << std::endl;
            queue_left_.erase(it_left);
            return tryGetNextPacket();
        }
        return std::nullopt; // 等待数据
    }

    // 3. 尝试匹配 Odom (应用时间偏移!)
    // VIO 时间 = 相机时间 + offset
    int64_t target_odom_time = t_left + offset_ns_;
    auto it_odom = findClosest(queue_odom_, target_odom_time, 20e6); // VIO 容差 20ms

    if (it_odom == queue_odom_.end()) {
        // 同理，等待 Odom
         if (queue_odom_.size() > 200 || queue_left_.size() > 50) {
             // Odom 彻底跟不上了，或者没数据
             // 这种情况下是否强制丢弃左帧？是的，没有 Pose 没法做 mapping
             queue_left_.erase(it_left);
             return tryGetNextPacket();
         }
         return std::nullopt;
    }

    // 4. 匹配成功，组装 Packet
    SyncedPacket packet;
    packet.timestamp_ns = t_left; // 统一用左目时间
    packet.left_disp = it_left->second;
    packet.right_disp = it_right->second;
    packet.odom = it_odom->second;

    // 简单提取 Pose (MVP 先不做插值，直接拿最近邻 Pose)
    // 实际生产建议在这里做线性插值 (Lerp)
    {
        auto reader = packet.odom.reader().getPose();
        auto pos = reader.getPosition();
        auto rot = reader.getOrientation();
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.translation() << pos.getX(), pos.getY(), pos.getZ();
        T.linear() = Eigen::Quaterniond(rot.getW(), rot.getX(), rot.getY(), rot.getZ()).toRotationMatrix();
        packet.interpolated_odom_pose = T;
    }

    // 5. 清理已使用的帧
    // 注意：Odom 频率高，不能只删一个，应该把 target_time 之前的旧数据都删了？
    // 稳妥起见，只从队列里移除当前匹配到的（如果是 map，移除 iterator）
    queue_left_.erase(it_left);
    queue_right_.erase(it_right);
    // queue_odom 不要随便删，因为下一个图像帧可能还需要插值用到这个 odom 区间
    // 这里先不删 odom，或者只删很久以前的
    
    // 清理非常老的 Odom
    auto it_old_odom = queue_odom_.begin();
    while(it_old_odom != queue_odom_.end() && it_old_odom->first < target_odom_time - 1e9) {
        it_old_odom = queue_odom_.erase(it_old_odom);
    }

    last_processed_time_ = t_left;
    return packet;
}

} // namespace trajectory