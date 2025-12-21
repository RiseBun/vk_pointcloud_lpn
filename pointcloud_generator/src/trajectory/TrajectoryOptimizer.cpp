#define MCAP_IMPLEMENTATION
#include <mcap/reader.hpp>

#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <iomanip>
#include <cmath>

#include <Eigen/Dense>
#include <Eigen/Geometry>

// 引入 vk_sdk 的 capnp 头文件 (根据你的实际路径调整)
#include <vk_sdk/capnp/odometry3d.capnp.h>
#include <vk_sdk/capnp/image.capnp.h>
#include <capnp/serialize.h>

// ---------------- 配置参数 ----------------
const int64_t CAM_TIME_OFFSET_NS = 8000000; // 8ms, 来自你的 saved_calib.json
const std::string VIO_TOPIC = "S1/vio_odom"; // 或者是 "S0/vio_odom"，请确认
const std::string LEFT_CAM_TOPIC = "S1/stereo1_l"; // 主时间轴参考
// ----------------------------------------

struct PoseData {
    int64_t timestamp_ns;
    Eigen::Vector3d t;
    Eigen::Quaterniond q;
};

// 辅助：从 capnp 读 pose 转 Eigen
// 注意：这里我们假设 mcap 里的 raw bytes 可以直接被 capnp 解析
// 如果你的 sdk 封装了 header，可能需要剥离 preamble
PoseData ExtractPose(const mcap::Message& msg) {
    // 这里是一个简化假设，实际 vk_sdk 可能有 Preamble
    // 如果直接解析失败，需要跳过 sizeof(vkc::Preamble) 字节
    // 通常 mcap payload 是纯 capnp 数据
    
    auto data_ptr = reinterpret_cast<const kj::byte*>(msg.data);
    auto data_size = msg.dataSize;

    // 假设没有 Preamble，直接是 capnp (视你的录制方式而定)
    // 如果有 Preamble，通常是 24 字节或更多，需要 offset
    
    // 使用 kj::ArrayPtr 包装数据
    kj::ArrayPtr<const capnp::word> wordPtr(
    reinterpret_cast<const capnp::word*>(data_ptr), 
    data_size / sizeof(capnp::word)
);
capnp::FlatArrayMessageReader reader(wordPtr);
    auto odom = reader.getRoot<vkc::Odometry3d>();
    
    PoseData p;
    // 获取单调时间 (Monotonic)
    p.timestamp_ns = odom.getHeader().getStampMonotonic(); 
    
    auto pos = odom.getPose().getPosition();
    auto rot = odom.getPose().getOrientation();
    
    p.t = Eigen::Vector3d(pos.getX(), pos.getY(), pos.getZ());
    p.q = Eigen::Quaterniond(rot.getW(), rot.getX(), rot.getY(), rot.getZ());
    p.q.normalize();
    
    return p;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./trajectory_optimizer <input.mcap> <output_poses.txt>" << std::endl;
        return 1;
    }

    std::string input_mcap = argv[1];
    std::string output_file = argv[2];

    std::cout << "Opening MCAP: " << input_mcap << std::endl;
    mcap::McapReader reader;
    auto status = reader.open(input_mcap);
    if (!status.ok()) {
        std::cerr << "Failed to open mcap: " << status.message << std::endl;
        return -1;
    }

    // 1. 读取所有 VIO 数据到 Map (自动按时间排序)
    std::cout << "Pass 1: Reading VIO data..." << std::endl;
    std::map<int64_t, PoseData> vio_map;
    
    auto onProblem = [](const mcap::Status& status) {
        std::cerr << "Mcap read error: " << status.message << std::endl;
    };

    mcap::ReadMessageOptions options;
    options.topicFilter = {VIO_TOPIC};
    
    for (const auto& msg : reader.readMessages(onProblem, options)) {
        try {
            PoseData p = ExtractPose(msg.message);
            vio_map[p.timestamp_ns] = p;
        } catch (...) {
            // 忽略解析错误
        }
    }
    std::cout << "Loaded " << vio_map.size() << " VIO poses." << std::endl;

    if (vio_map.empty()) {
        std::cerr << "Error: No VIO data found in topic " << VIO_TOPIC << std::endl;
        return -1;
    }

    // 2. 遍历图像时间戳，插值 VIO
    std::cout << "Pass 2: Processing Images and Interpolating..." << std::endl;
    std::ofstream outfile(output_file);
    outfile << std::fixed << std::setprecision(9); // 保证时间戳精度
    
    // 我们不需要解析图像内容，只需要 Message LogTime 或 Header Stamp
    // 为了准确，我们应该解析 Header Stamp，但为了速度，如果 LogTime 接近也可以
    // 这里最好解析 Header。
    
    options.topicFilter = {LEFT_CAM_TOPIC};
    
    int frame_count = 0;
    
    for (const auto& msg : reader.readMessages(onProblem, options)) {
        // 解析图像 Header 获取时间戳
        auto data_ptr = reinterpret_cast<const kj::byte*>(msg.message.data);
        auto data_size = msg.message.dataSize;
        kj::ArrayPtr<const capnp::word> wordPtr(
    reinterpret_cast<const capnp::word*>(data_ptr), 
    data_size / sizeof(capnp::word)
);
capnp::FlatArrayMessageReader reader(wordPtr);
        auto image = capnp_reader.getRoot<vkc::Image>();
        
        int64_t img_time_ns = image.getHeader().getStampMonotonic();
        
        // 核心修正逻辑：
        // 图像是在 T 时刻拍摄的。
        // 但由于 VIO 和相机的延迟差异 (cam_time_offset_ns)，
        // 实际上对应 T + 8ms 时刻的 VIO 状态。
        int64_t query_time_ns = img_time_ns + CAM_TIME_OFFSET_NS;

        // 在 vio_map 中寻找 query_time_ns 左右的数据
        auto it_upper = vio_map.lower_bound(query_time_ns);
        
        if (it_upper == vio_map.begin() || it_upper == vio_map.end()) {
            // 时间戳越界（比如图像开始录制了但VIO还没来，或者VIO断了）
            // 跳过此帧
            continue;
        }
        
        auto it_lower = std::prev(it_upper);
        
        // 执行插值
        int64_t t1 = it_lower->first;
        int64_t t2 = it_upper->first;
        double alpha = (double)(query_time_ns - t1) / (double)(t2 - t1);
        
        Eigen::Vector3d pos_interp = it_lower->second.t * (1.0 - alpha) + it_upper->second.t * alpha;
        Eigen::Quaterniond q_interp = it_lower->second.q.slerp(alpha, it_upper->second.q);
        
        // 输出格式：
        // timestamp tx ty tz qx qy qz qw
        // 注意：timestamp 输出的是 img_time_ns (为了让 generator 能匹配上图像)
        // 但 pose 是修正后的 pose
        
        outfile << img_time_ns << " "
                << pos_interp.x() << " " << pos_interp.y() << " " << pos_interp.z() << " "
                << q_interp.x() << " " << q_interp.y() << " " << q_interp.z() << " " << q_interp.w() 
                << std::endl;
                
        frame_count++;
    }

    std::cout << "Finished. Generated " << frame_count << " optimized poses." << std::endl;
    outfile.close();
    
    return 0;
}