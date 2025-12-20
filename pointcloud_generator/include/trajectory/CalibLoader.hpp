#pragma once
#include <string>
#include <Eigen/Dense>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>

namespace trajectory {
struct JsonPose {
    double px, py, pz, qx, qy, qz, qw;
    template <class Archive> void serialize(Archive &ar) {
        ar(cereal::make_nvp("px",px), cereal::make_nvp("py",py), cereal::make_nvp("pz",pz),
           cereal::make_nvp("qx",qx), cereal::make_nvp("qy",qy), cereal::make_nvp("qz",qz), cereal::make_nvp("qw",qw));
    }
};
struct CalibData {
    JsonPose T_imu_body;
    std::vector<JsonPose> T_imu_cam;
    int64_t cam_time_offset_ns;
    template <class Archive> void serialize(Archive &ar) {
        ar(cereal::make_nvp("T_imu_body", T_imu_body), cereal::make_nvp("T_imu_cam", T_imu_cam),
           cereal::make_nvp("cam_time_offset_ns", cam_time_offset_ns));
    }
};
struct CalibRoot { CalibData value0; template <class Archive> void serialize(Archive &ar) { ar(cereal::make_nvp("value0", value0)); } };

class CalibLoader {
public:
    bool load(const std::string& path);
    int64_t getCamTimeOffsetNs() const { return data_.value0.cam_time_offset_ns; }
private:
    CalibRoot data_;
};
}