#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>
#include <vector>
#include <cmath>
#include <cstring> 
#include <Eigen/Dense>
#include <Eigen/Geometry>

#define MCAP_IMPLEMENTATION
#include <mcap/reader.hpp>

#include "trajectory/CalibLoader.hpp"
#include <vk_sdk/capnp/odometry3d.capnp.h>
#include <vk_sdk/capnp/image.capnp.h>
#include <capnp/serialize.h>

using namespace trajectory;

// 辅助函数：对齐内存
std::vector<uint64_t> align_data(const std::byte* data, size_t size) {
    size_t word_count = (size + 7) / 8;
    std::vector<uint64_t> aligned_buffer(word_count);
    std::memcpy(aligned_buffer.data(), data, size);
    return aligned_buffer;
}

void process_offline_mcap(const std::string& mcap_in, const std::string& json_in, const std::string& txt_out) {
    // 1. 加载标定
    CalibLoader calib;
    if (!calib.load(json_in)) {
        throw std::runtime_error("Could not load calibration JSON");
    }
    
    int64_t offset_ns = calib.getCamTimeOffsetNs();
    
    std::cout << "[Step 1] Calibration loaded." << std::endl;
    std::cout << "  - Hardware Sync Offset: " << offset_ns << " ns" << std::endl;
    
    // 检查 Offset 是否异常大（如果是 Unix 偏移，这里会报警）
    if (std::abs(offset_ns) > 100000000000LL) { // > 100秒
        std::cerr << "\033[1;31m[WARNING] Huge offset detected! This might break Monotonic alignment.\033[0m" << std::endl;
    }

    // 2. 打开 MCAP
    mcap::McapReader reader;
    auto status = reader.open(mcap_in);
    if (!status.ok()) throw std::runtime_error("Failed to open MCAP");

    // [重要] 确保 Topic 名字和你录制的一致
    const std::string VIO_TOPIC = "S1/vio_odom"; 
    const std::string LEFT_CAM_TOPIC = "S1/stereo1_l";

    // 3. 读取 VIO (World -> IMU)
    std::cout << "[Step 2] Reading VIO poses from " << VIO_TOPIC << "..." << std::endl;
    std::map<int64_t, Eigen::Isometry3d> vio_map;
    
    mcap::ReadMessageOptions options;
    options.topicFilter = [&](std::string_view topic) { return topic == VIO_TOPIC; };
    
    for (const auto& msg : reader.readMessages([](const auto&){}, options)) {
        auto aligned_buf = align_data(msg.message.data, msg.message.dataSize);
        kj::ArrayPtr<const capnp::word> wordPtr(reinterpret_cast<const capnp::word*>(aligned_buf.data()), aligned_buf.size());
        capnp::FlatArrayMessageReader cap_reader(wordPtr);
        auto odom = cap_reader.getRoot<vkc::Odometry3d>();
        
        auto pos = odom.getPose().getPosition();
        auto rot = odom.getPose().getOrientation();
        
        Eigen::Isometry3d T_w_i = Eigen::Isometry3d::Identity();
        T_w_i.translation() << pos.getX(), pos.getY(), pos.getZ();
        T_w_i.linear() = Eigen::Quaterniond(rot.getW(), rot.getX(), rot.getY(), rot.getZ()).toRotationMatrix();
        
        // 存储 Monotonic 时间
        vio_map[odom.getHeader().getStampMonotonic()] = T_w_i;
    }
    std::cout << "  - Loaded " << vio_map.size() << " VIO messages." << std::endl;

    if (vio_map.empty()) throw std::runtime_error("No VIO data found! Check topic name.");

    // 4. 对齐并输出 (输出 T_world_body)
    std::cout << "[Step 3] Syncing to Image Frames (" << LEFT_CAM_TOPIC << ")..." << std::endl;
    std::ofstream out_file(txt_out);
    out_file << std::fixed << std::setprecision(9);

    options.topicFilter = [&](std::string_view topic) { return topic == LEFT_CAM_TOPIC; };
    
    int count = 0;
    for (const auto& msg : reader.readMessages([](const auto&){}, options)) {
        auto aligned_buf = align_data(msg.message.data, msg.message.dataSize);
        kj::ArrayPtr<const capnp::word> wordPtr(reinterpret_cast<const capnp::word*>(aligned_buf.data()), aligned_buf.size());
        capnp::FlatArrayMessageReader cap_reader(wordPtr);
        auto img = cap_reader.getRoot<vkc::Image>();
        
        int64_t img_t = img.getHeader().getStampMonotonic();
        int64_t query_t = img_t + offset_ns; // 加上硬件同步偏差

        auto it = vio_map.lower_bound(query_t);
        if (it == vio_map.end() || it == vio_map.begin()) continue; 

        auto it_prev = std::prev(it);
        
        // 插值
        double alpha = (double)(query_t - it_prev->first) / (double)(it->first - it_prev->first);
        if (std::abs(it->first - it_prev->first) > 100000000) continue; // Gap > 100ms skip

        Eigen::Vector3d t_interp = it_prev->second.translation() * (1.0 - alpha) + it->second.translation() * alpha;
        Eigen::Quaterniond q1(it_prev->second.linear());
        Eigen::Quaterniond q2(it->second.linear());
        Eigen::Quaterniond q_interp = q1.slerp(alpha, q2);
        
        // 写入文件: timestamp tx ty tz qx qy qz qw
        out_file << (double)img_t * 1e-9 << " " 
                 << t_interp.x() << " " << t_interp.y() << " " << t_interp.z() << " "
                 << q_interp.x() << " " << q_interp.y() << " " << q_interp.z() << " " << q_interp.w() << "\n";
        count++;
    }
    std::cout << "[Success] Generated " << count << " poses to " << txt_out << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./trajectory_optimizer <input.mcap> <calib.json> <output_poses.txt>" << std::endl;
        return -1;
    }
    try {
        process_offline_mcap(argv[1], argv[2], argv[3]);
    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}