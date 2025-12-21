
#include <mcap/reader.hpp>

#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <iomanip>
#include <cmath>
#include <string_view>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <vk_sdk/capnp/odometry3d.capnp.h>
#include <vk_sdk/capnp/image.capnp.h>
#include <capnp/serialize.h>

const int64_t CAM_TIME_OFFSET_NS = 8000000; 
const std::string VIO_TOPIC = "S1/vio_odom"; 
const std::string LEFT_CAM_TOPIC = "S1/stereo1_l"; 

struct PoseData {
    int64_t timestamp_ns;
    Eigen::Vector3d t;
    Eigen::Quaterniond q;
};

PoseData ExtractPose(const mcap::Message& msg) {
    // 修复：将字节指针转换为 capnp::word 指针以满足对齐要求
    kj::ArrayPtr<const capnp::word> wordPtr(
        reinterpret_cast<const capnp::word*>(msg.data), 
        msg.dataSize / sizeof(capnp::word)
    );
    capnp::FlatArrayMessageReader reader(wordPtr);
    auto odom = reader.getRoot<vkc::Odometry3d>();
    
    PoseData p;
    p.timestamp_ns = odom.getHeader().getStampMonotonic(); 
    
    auto pos = odom.getPose().getPosition();
    auto rot = odom.getPose().getOrientation();
    
    p.t = Eigen::Vector3d(pos.getX(), pos.getY(), pos.getZ());
    p.q = Eigen::Quaterniond(rot.getW(), rot.getX(), rot.getY(), rot.getZ());
    p.q.normalize();
    
    return p;
}

