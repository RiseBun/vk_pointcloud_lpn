#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>
#include <Eigen/Dense>

#define MCAP_IMPLEMENTATION
#include <mcap/reader.hpp>

#include "trajectory/CalibLoader.hpp"
#include <vk_sdk/capnp/odometry3d.capnp.h>
#include <vk_sdk/capnp/image.capnp.h>
#include <capnp/serialize.h>

using namespace trajectory;

// 离线处理函数声明
void process_offline_mcap(const std::string& mcap_in, const std::string& json_in, const std::string& txt_out);

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./trajectory_optimizer <input.mcap> <saved_calib.json> <output_poses.txt>" << std::endl;
        return -1;
    }
    
    std::string mcap_path = argv[1];
    std::string calib_path = argv[2];
    std::string output_path = argv[3];

    try {
        process_offline_mcap(mcap_path, calib_path, output_path);
    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}

void process_offline_mcap(const std::string& mcap_in, const std::string& json_in, const std::string& txt_out) {
    // 1. 加载标定 (获取 8ms 偏移量)
    CalibLoader calib;
    if (!calib.load(json_in)) {
        throw std::runtime_error("Could not load calibration JSON: " + json_in);
    }
    int64_t offset_ns = calib.getCamTimeOffsetNs();
    std::cout << "[1/3] Calibration loaded. Time offset: " << offset_ns << " ns" << std::endl;

    // 2. 读取 VIO 轨迹
    mcap::McapReader reader;
    if (!reader.open(mcap_in).ok()) throw std::runtime_error("Failed to open MCAP: " + mcap_in);

    const std::string VIO_TOPIC = "S1/vio_odom"; 
    const std::string LEFT_CAM_TOPIC = "S1/stereo1_l";

    std::map<int64_t, Eigen::Isometry3d> vio_map;
    mcap::ReadMessageOptions options;
    options.topicFilter = [&](std::string_view topic) { return topic == VIO_TOPIC; };
    
    for (const auto& msg : reader.readMessages([](const auto&){}, options)) {
        kj::ArrayPtr<const capnp::word> wordPtr(reinterpret_cast<const capnp::word*>(msg.message.data), msg.message.dataSize / sizeof(capnp::word));
        capnp::FlatArrayMessageReader cap_reader(wordPtr);
        auto odom = cap_reader.getRoot<vkc::Odometry3d>();
        
        auto pos = odom.getPose().getPosition();
        auto rot = odom.getPose().getOrientation();
        Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
        T.translation() << pos.getX(), pos.getY(), pos.getZ();
        T.linear() = Eigen::Quaterniond(rot.getW(), rot.getX(), rot.getY(), rot.getZ()).toRotationMatrix();
        
        vio_map[odom.getHeader().getStampMonotonic()] = T;
    }
    std::cout << "[2/3] Read " << vio_map.size() << " VIO poses." << std::endl;

    // 3. 对齐图像时间戳并插值输出
    std::ofstream out_file(txt_out);
    out_file << std::fixed << std::setprecision(9);
    options.topicFilter = [&](std::string_view topic) { return topic == LEFT_CAM_TOPIC; };
    
    int count = 0;
    for (const auto& msg : reader.readMessages([](const auto&){}, options)) {
        kj::ArrayPtr<const capnp::word> wordPtr(reinterpret_cast<const capnp::word*>(msg.message.data), msg.message.dataSize / sizeof(capnp::word));
        capnp::FlatArrayMessageReader cap_reader(wordPtr);
        auto img = cap_reader.getRoot<vkc::Image>();
        
        int64_t img_t = img.getHeader().getStampMonotonic();
        int64_t query_t = img_t + offset_ns;

        auto it = vio_map.lower_bound(query_t);
        if (it == vio_map.end() || it == vio_map.begin()) continue;

        auto it_prev = std::prev(it);
        double alpha = (double)(query_t - it_prev->first) / (it->first - it_prev->first);
        
        Eigen::Vector3d t = it_prev->second.translation() * (1.0 - alpha) + it->second.translation() * alpha;
        Eigen::Quaterniond q = Eigen::Quaterniond(it_prev->second.linear()).slerp(alpha, Eigen::Quaterniond(it->second.linear()));

        out_file << img_t << " " << t.x() << " " << t.y() << " " << t.z() << " "
                 << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        count++;
    }
    std::cout << "[3/3] Success! Saved " << count << " aligned poses to " << txt_out << std::endl;
}