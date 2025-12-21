#pragma once
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>

namespace trajectory {

// 1. 对应 JSON 中的坐标 (px, py, ...)
struct JsonPose {
    double px, py, pz, qx, qy, qz, qw;
    
    template <class Archive> 
    void serialize(Archive &ar) {
        ar(cereal::make_nvp("px", px), cereal::make_nvp("py", py), cereal::make_nvp("pz", pz),
           cereal::make_nvp("qx", qx), cereal::make_nvp("qy", qy), cereal::make_nvp("qz", qz), cereal::make_nvp("qw", qw));
    }
};

// 2. 对应 value0 内部的数据字段
struct CalibData {
    JsonPose T_imu_body;
    std::vector<JsonPose> T_imu_cam;
    int64_t cam_time_offset_ns;

    template <class Archive> 
    void serialize(Archive &ar) {
        // Cereal 会在当前层级寻找这些名字
        ar(cereal::make_nvp("T_imu_body", T_imu_body), 
           cereal::make_nvp("T_imu_cam", T_imu_cam),
           cereal::make_nvp("cam_time_offset_ns", cam_time_offset_ns));
    }
};

// 3. 加载器类
class CalibLoader {
public:
    bool load(const std::string& path);
    int64_t getCamTimeOffsetNs() const { return data_.cam_time_offset_ns; }
private:
    CalibData data_;
};

}
