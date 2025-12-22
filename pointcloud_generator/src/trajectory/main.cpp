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
    
    // 获取时间偏移
    int64_t offset_ns = calib.getCamTimeOffsetNs();
    
    // [调试信息] 获取 IMU 到 左目相机 的外参 (T_imu_cam)
    auto& json_T = calib.data_.T_imu_cam[0]; 
    Eigen::Isometry3d T_i_c = Eigen::Isometry3d::Identity();
    T_i_c.translation() << json_T.px, json_T.py, json_T.pz;
    T_i_c.linear() = Eigen::Quaterniond(json_T.qw, json_T.qx, json_T.qy, json_T.qz).toRotationMatrix();

    std::cout << "[Step 1] Calibration loaded." << std::endl;
    std::cout << "  - Time Offset: " << offset_ns << " ns" << std::endl;
    
    // [Phase 1 修改提示] 
    std::cout << "  - Mode: Exporting VIO POSE (World->Body/IMU) directly." << std::endl;
    std::cout << "    (Downstream point_cloud_gen will handle Body->Cam transform)" << std::endl;

    // [Frame Semantics Probe] 打印相机名称和外参以人工核对 (保留你的要求)
    std::cout << "\n[Frame Semantics Probe]" << std::endl;
    for (size_t i = 0; i < calib.data_.cam_names.size(); ++i) {
        std::cout << "  - Cam[" << i << "]: " << calib.data_.cam_names[i] << std::endl;
        if (i == 0) { 
            std::cout << "    T_imu_cam[0] Translation: [" 
                      << json_T.px << ", " << json_T.py << ", " << json_T.pz << "]" << std::endl;
        }
    }
    std::cout << "------------------------------------------------\n" << std::endl;

    // 2. 打开 MCAP
    mcap::McapReader reader;
    auto status = reader.open(mcap_in);
    if (!status.ok()) throw std::runtime_error("Failed to open MCAP");

    const std::string VIO_TOPIC = "S1/vio_odom"; 
    const std::string LEFT_CAM_TOPIC = "S1/stereo1_l";

    // 3. 读取 VIO (World -> IMU)
    std::cout << "[Step 2] Reading VIO poses (T_world_imu)..." << std::endl;
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
        
        // 注意：这里存的是 stampMonotonic，非常正确，保持不变
        vio_map[odom.getHeader().getStampMonotonic()] = T_w_i;
    }
    std::cout << "  - Found " << vio_map.size() << " VIO messages." << std::endl;

    if (vio_map.empty()) throw std::runtime_error("No VIO data found!");

    // 4. 对齐并输出 (输出 T_world_body)
    std::cout << "[Step 3] Syncing and converting to T_world_body..." << std::endl;
    std::ofstream out_file(txt_out);
    out_file << std::fixed << std::setprecision(9);

    options.topicFilter = [&](std::string_view topic) { return topic == LEFT_CAM_TOPIC; };
    
    // 用于探针的静态变量
    static bool first_img = true;
    static int64_t last_img_t = 0;
    static int interval_check_count = 0;

    int count = 0;
    for (const auto& msg : reader.readMessages([](const auto&){}, options)) {
        auto aligned_buf = align_data(msg.message.data, msg.message.dataSize);
        kj::ArrayPtr<const capnp::word> wordPtr(reinterpret_cast<const capnp::word*>(aligned_buf.data()), aligned_buf.size());
        capnp::FlatArrayMessageReader cap_reader(wordPtr);
        auto img = cap_reader.getRoot<vkc::Image>();
        
        int64_t img_t = img.getHeader().getStampMonotonic();
        
        // [时间对齐协议]
        // 图像时间 img_t 对应的 IMU 时间是 img_t + offset
        int64_t query_t = img_t + offset_ns;

        // ==========================================
        // [Time Unit Probe] 必须在 loop 内部 (保留你的要求)
        // ==========================================
        if (first_img) {
            std::cout << "\n[Time Unit Probe] First Image Frame:" << std::endl;
            std::cout << "  - Raw Monotonic: " << img_t << std::endl;
            std::cout << "  - Offset Used:   " << offset_ns << std::endl;
            std::cout << "  - Query Time:    " << query_t << std::endl;
            
            // 粗略判断量级
            if (img_t > 1e12 && img_t < 1e16) {
                std::cout << "  -> Hypothesis: Unit is NANOSECONDS (ns)." << std::endl;
            } else if (img_t > 1e9 && img_t < 1e12) {
                std::cout << "  -> Hypothesis: Unit is MICROSECONDS (us)." << std::endl;
            } else {
                std::cout << "  -> Hypothesis: Unit is UNKNOWN." << std::endl;
            }
            
            last_img_t = img_t;
            first_img = false;
        } else {
            // 打印帧间隔，进一步验证
            if (interval_check_count < 5) {
                int64_t delta = img_t - last_img_t;
                std::cout << "  - Frame Interval: " << delta << " (raw units)";
                if (delta > 30000000 && delta < 100000000) std::cout << " -> Looks like 33ms-100ms in NS";
                std::cout << std::endl;
                last_img_t = img_t;
                interval_check_count++;
            }
        }
        // ==========================================

        auto it = vio_map.lower_bound(query_t);
        if (it == vio_map.end() || it == vio_map.begin()) continue; 

        auto it_prev = std::prev(it);
        
        // 插值计算 T_w_i (World -> IMU)
        double alpha = (double)(query_t - it_prev->first) / (double)(it->first - it_prev->first);
        
        Eigen::Vector3d t_interp = it_prev->second.translation() * (1.0 - alpha) + it->second.translation() * alpha;
        Eigen::Quaterniond q1(it_prev->second.linear());
        Eigen::Quaterniond q2(it->second.linear());
        Eigen::Quaterniond q_interp = q1.slerp(alpha, q2);
        
        // -----------------------------------------------------
        // [Phase 1 核心修改]
        // 之前你是: T_w_c = T_w_i * T_i_c; (导出相机位姿)
        // 现在改为: 直接导出 T_w_i (导出 Body/IMU 位姿)
        // -----------------------------------------------------
        
        Eigen::Vector3d t_out = t_interp; 
        Eigen::Quaterniond q_out = q_interp;

        // 输出格式：timestamp(monotonic_sec) tx ty tz qx qy qz qw
        out_file << (double)img_t * 1e-9 << " " 
                 << t_out.x() << " " << t_out.y() << " " << t_out.z() << " "
                 << q_out.x() << " " << q_out.y() << " " << q_out.z() << " " << q_out.w() << "\n";
        count++;
    }
    std::cout << "[Success] Generated " << count << " BODY poses to " << txt_out << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: ./trajectory_optimizer <input.mcap> <saved_calib.json> <output_poses.txt>" << std::endl;
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