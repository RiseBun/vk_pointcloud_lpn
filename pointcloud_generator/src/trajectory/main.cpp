#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>
#include <vector>
#include <cmath>
#include <cstring> // for memcpy
#include <Eigen/Dense>
#include <Eigen/Geometry>

// 必须定义此宏以引入 MCAP 实现
#define MCAP_IMPLEMENTATION
#include <mcap/reader.hpp>

#include "trajectory/CalibLoader.hpp"
#include <vk_sdk/capnp/odometry3d.capnp.h>
#include <vk_sdk/capnp/image.capnp.h>
#include <capnp/serialize.h>

using namespace trajectory;

void process_offline_mcap(const std::string& mcap_in, const std::string& json_in, const std::string& txt_out);

// 辅助函数：将未对齐的 mcap 数据拷贝到对齐的 buffer 中
std::vector<uint64_t> align_data(const std::byte* data, size_t size) {
    // 计算需要的 64位(8字节) 字数，向上取整
    size_t word_count = (size + 7) / 8;
    std::vector<uint64_t> aligned_buffer(word_count);
    // 拷贝数据到对齐的 vector 中
    std::memcpy(aligned_buffer.data(), data, size);
    return aligned_buffer;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./trajectory_optimizer <input.mcap> <saved_calib.json> <output_poses.txt>" << std::endl;
        return -1;
    }
    
    std::string mcap_path = argv[1];
    std::string calib_path = argv[2];
    std::string output_path = argv[3];

    std::cout << "--- Trajectory Optimizer (Offline / Aligned) ---" << std::endl;
    std::cout << "Input MCAP: " << mcap_path << std::endl;

    try {
        process_offline_mcap(mcap_path, calib_path, output_path);
    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

void process_offline_mcap(const std::string& mcap_in, const std::string& json_in, const std::string& txt_out) {
    // 1. 加载标定
    CalibLoader calib;
    if (!calib.load(json_in)) {
        throw std::runtime_error("Could not load calibration JSON: " + json_in);
    }
    int64_t offset_ns = calib.getCamTimeOffsetNs();
    std::cout << "[Step 1] Calibration loaded. Time Offset: " << offset_ns << " ns" << std::endl;

    // 2. 打开 MCAP
    mcap::McapReader reader;
    auto status = reader.open(mcap_in);
    if (!status.ok()) {
        throw std::runtime_error("Failed to open MCAP: " + status.message);
    }

    const std::string VIO_TOPIC = "S1/vio_odom"; 
    const std::string LEFT_CAM_TOPIC = "S1/stereo1_l";

    // 3. 读取 VIO (带对齐处理)
    std::cout << "[Step 2] Reading VIO poses..." << std::endl;
    std::map<int64_t, Eigen::Isometry3d> vio_map;
    
    mcap::ReadMessageOptions options;
    options.topicFilter = [&](std::string_view topic) { return topic == VIO_TOPIC; };
    
    for (const auto& msg : reader.readMessages([](const auto&){}, options)) {
        // --- 核心修正：内存拷贝以确保对齐 ---
        auto aligned_buf = align_data(msg.message.data, msg.message.dataSize);
        kj::ArrayPtr<const capnp::word> wordPtr(
            reinterpret_cast<const capnp::word*>(aligned_buf.data()), 
            aligned_buf.size()
        );
        // -------------------------------------
        
        capnp::FlatArrayMessageReader cap_reader(wordPtr);
        auto odom = cap_reader.getRoot<vkc::Odometry3d>();
        
        auto pos = odom.getPose().getPosition();
        auto rot = odom.getPose().getOrientation();
        
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.translation() << pos.getX(), pos.getY(), pos.getZ();
        T.linear() = Eigen::Quaterniond(rot.getW(), rot.getX(), rot.getY(), rot.getZ()).toRotationMatrix();
        
        vio_map[odom.getHeader().getStampMonotonic()] = T;
    }
    std::cout << "  - Found " << vio_map.size() << " VIO messages." << std::endl;

    if (vio_map.empty()) {
        throw std::runtime_error("No VIO data found! Check topic name.");
    }

    // 4. 读取图像并插值 (带对齐处理)
    std::cout << "[Step 3] Syncing..." << std::endl;
    std::ofstream out_file(txt_out);
    out_file << std::fixed << std::setprecision(9);

    options.topicFilter = [&](std::string_view topic) { return topic == LEFT_CAM_TOPIC; };
    
    int count = 0;
    for (const auto& msg : reader.readMessages([](const auto&){}, options)) {
        // --- 核心修正：内存拷贝以确保对齐 ---
        auto aligned_buf = align_data(msg.message.data, msg.message.dataSize);
        kj::ArrayPtr<const capnp::word> wordPtr(
            reinterpret_cast<const capnp::word*>(aligned_buf.data()), 
            aligned_buf.size()
        );
        // -------------------------------------

        capnp::FlatArrayMessageReader cap_reader(wordPtr);
        auto img = cap_reader.getRoot<vkc::Image>();
        
        int64_t img_t = img.getHeader().getStampMonotonic();
        int64_t query_t = img_t + offset_ns;

        auto it = vio_map.lower_bound(query_t);
        if (it == vio_map.end() || it == vio_map.begin()) continue; 

        auto it_prev = std::prev(it);
        double alpha = (double)(query_t - it_prev->first) / (double)(it->first - it_prev->first);
        
        Eigen::Vector3d t = it_prev->second.translation() * (1.0 - alpha) + it->second.translation() * alpha;
        Eigen::Quaterniond q1(it_prev->second.linear());
        Eigen::Quaterniond q2(it->second.linear());
        Eigen::Quaterniond q = q1.slerp(alpha, q2);

        out_file << img_t << " " 
                 << t.x() << " " << t.y() << " " << t.z() << " "
                 << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        count++;
    }
    std::cout << "[Success] Saved " << count << " poses to " << txt_out << std::endl;
}
