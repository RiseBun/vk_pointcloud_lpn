#pragma once

#include <string>
#include <vector>
#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>

namespace trajectory {

// 定义 Pose 结构对应 JSON 里的 T_imu_cam 元素
struct JsonPose {
    double px, py, pz;
    double qx, qy, qz, qw;

    template <class Archive>
    void serialize(Archive & ar) {
        ar(cereal::make_nvp("px", px), 
           cereal::make_nvp("py", py), 
           cereal::make_nvp("pz", pz),
           cereal::make_nvp("qx", qx), 
           cereal::make_nvp("qy", qy), 
           cereal::make_nvp("qz", qz), 
           cereal::make_nvp("qw", qw));
    }
};

// 定义核心数据结构
struct CalibData {
    std::vector<JsonPose> T_imu_cam;
    std::vector<std::string> cam_names; // [新增] 补充 cam_names 字段
    long long cam_time_offset_ns;

    // 序列化函数，对应 JSON 里的字段名
    template <class Archive>
    void serialize(Archive & ar) {
        // Cereal 会自动读取 JSON 中对应的字段
        // 使用 make_optional_nvp 以防某些字段不存在时不报错（可选，这里用标准 nvp 即可）
        ar(cereal::make_nvp("T_imu_cam", T_imu_cam),
           cereal::make_nvp("cam_names", cam_names),
           cereal::make_nvp("cam_time_offset_ns", cam_time_offset_ns));
    }
};

class CalibLoader {
public:
    bool load(const std::string& path);

    int64_t getCamTimeOffsetNs() const {
        return data_.cam_time_offset_ns;
    }

    // [修改] 将 data_ 改为 public，以便 main.cpp 可以直接访问验证
    CalibData data_; 
};

} // namespace trajectory