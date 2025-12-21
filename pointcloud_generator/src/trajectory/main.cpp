#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <fstream>
#include <ecal/ecal.h>
#include <ecal/msg/capnproto/subscriber.h>

#include "trajectory/CalibLoader.hpp"
#include "trajectory/TimeSync.hpp"
// 这里需要根据你的 vk_sdk 路径 include 对应的 capnp 头文件
#include <vk_sdk/capnp/disparity.capnp.h>
#include <vk_sdk/capnp/odometry3d.capnp.h>

using namespace trajectory;

std::atomic<bool> g_running{true};

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./trajectory_optimizer <saved_calib.json> <output_poses.txt>" << std::endl;
        return -1;
    }
    
    std::string calib_path = argv[1];
    std::string output_path = argv[2];

    // 1. 加载标定
    CalibLoader calib;
    if (!calib.load(calib_path)) {
        return -1;
    }
    std::cout << "Calibration loaded. Time offset: " << calib.getCamTimeOffsetNs() << std::endl;

    // 2. 初始化 eCAL
    eCAL::Initialize(argc, argv, "TrajectoryOptimizer");
    
    // 3. 初始化时间同步
    TimeSync sync(calib.getCamTimeOffsetNs());

    // 4. 设置 eCAL 订阅 (直接用 Capnp Subscriber)
    // 假设 Topic 名字如下，请根据实际情况修改
    eCAL::capnproto::Subscriber<vkc::Disparity> sub_left("S1/stereo1_l/disparity");
    eCAL::capnproto::Subscriber<vkc::Disparity> sub_right("S1/stereo2_r/disparity");
    eCAL::capnproto::Subscriber<vkc::Odometry3d> sub_odom("S1/vio_odom");

    auto on_left = [&](const eCAL::capnproto::Msg<vkc::Disparity>& msg) {
        // 这里 msg 已经是 capnp reader 封装，需要转成 vkc::Shared 或者直接把数据拷进去
        // 为了配合上面的 TimeSync 接口 (vkc::Shared)，可能需要一点适配代码
        // 这里为了 MVP 演示，假设有一个转换函数
        // sync.addLeftDisp(convert(msg)); 
        // *实际写代码时，这一步如果是 vk_sdk 的 Receiver 就会自动处理好 Shared 指针*
    };
    
    // 如果不想依赖 vk_sdk 的 Receiver 封装，可以直接修改 TimeSync 接收 capnp Reader
    
    std::cout << "Waiting for data..." << std::endl;

    // 5. 主循环处理
    std::ofstream out_file(output_path);
    out_file << "# t_ns tx ty tz qx qy qz qw" << std::endl;

    while (eCAL::Ok() && g_running) {
        // 尝试获取同步帧
        auto packet_opt = sync.tryGetNextPacket();
        
        if (packet_opt) {
            auto& pkt = *packet_opt;
            if (pkt.interpolated_odom_pose) {
                // 导出 Pose
                auto p = pkt.interpolated_odom_pose.value();
                Eigen::Vector3d t = p.translation();
                Eigen::Quaterniond q(p.linear());
                
                out_file << pkt.timestamp_ns << " " 
                         << t.x() << " " << t.y() << " " << t.z() << " "
                         << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
                         
                std::cout << "Synced frame at " << pkt.timestamp_ns << std::endl;
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    eCAL::Finalize();
    return 0;
}