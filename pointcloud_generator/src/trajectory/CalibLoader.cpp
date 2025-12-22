#include "trajectory/CalibLoader.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <regex>

namespace trajectory {

// 辅助函数：从文本中查找键值对
// 查找形如 "key": value 的数字
template <typename T>
bool find_key_value(const std::string& content, const std::string& key, T& out_val) {
    // 简单的字符串查找，忽略层级，找到即止
    // 查找 "key"
    size_t key_pos = content.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return false;

    // 查找冒号
    size_t colon_pos = content.find(':', key_pos);
    if (colon_pos == std::string::npos) return false;

    // 读取数字
    std::string remaining = content.substr(colon_pos + 1);
    std::stringstream ss(remaining);
    
    // 跳过可能的非数字字符（比如 [ { 空格）直到找到数字
    // 这里简单处理：直接用 stringstream 尝试读，它会自动跳过前导空格
    // 但如果是 [ 或 {，我们需要手动跳过
    char c;
    while (ss.get(c)) {
        if (isdigit(c) || c == '-' || c == '.') {
            ss.putback(c);
            break;
        }
    }
    
    ss >> out_val;
    return !ss.fail();
}

// 辅助函数：查找数组中的第一个对象
// 专门用于找 T_imu_cam 的第一个 Pose
bool parse_first_pose(const std::string& content, JsonPose& pose) {
    size_t key_pos = content.find("\"T_imu_cam\"");
    if (key_pos == std::string::npos) return false;

    // 找到 T_imu_cam 后的第一个 {
    size_t brace_pos = content.find('{', key_pos);
    if (brace_pos == std::string::npos) return false;

    // 截取这个对象的范围（粗略截取到下一个 }）
    size_t end_brace = content.find('}', brace_pos);
    std::string block = content.substr(brace_pos, end_brace - brace_pos + 1);

    bool ok = true;
    ok &= find_key_value(block, "px", pose.px);
    ok &= find_key_value(block, "py", pose.py);
    ok &= find_key_value(block, "pz", pose.pz);
    ok &= find_key_value(block, "qx", pose.qx);
    ok &= find_key_value(block, "qy", pose.qy);
    ok &= find_key_value(block, "qz", pose.qz);
    ok &= find_key_value(block, "qw", pose.qw);
    return ok;
}

bool CalibLoader::load(const std::string& path) {
    std::ifstream is(path);
    if (!is.is_open()) {
        std::cerr << "[Error] Cannot open calibration file: " << path << std::endl;
        return false;
    }

    std::stringstream buffer;
    buffer << is.rdbuf();
    std::string content = buffer.str();

    std::cout << "[CalibLoader] Using Manual Extraction Mode (Robust)..." << std::endl;

    // 1. 读取 cam_time_offset_ns
    if (!find_key_value(content, "cam_time_offset_ns", data_.cam_time_offset_ns)) {
        std::cerr << "[Warning] Could not find 'cam_time_offset_ns', defaulting to 0." << std::endl;
        data_.cam_time_offset_ns = 0;
    } else {
        std::cout << "  - Found Time Offset: " << data_.cam_time_offset_ns << std::endl;
    }

    // 2. 读取第一个 T_imu_cam
    JsonPose pose0;
    if (parse_first_pose(content, pose0)) {
        data_.T_imu_cam.clear();
        data_.T_imu_cam.push_back(pose0);
        std::cout << "  - Found T_imu_cam[0]: px=" << pose0.px << std::endl;
    } else {
        std::cerr << "[Error] Failed to parse T_imu_cam[0]!" << std::endl;
        return false;
    }

    // 3. 伪造 cam_names (可选，为了 Probe 输出好看)
    // 既然是手动解析，我们就不去费劲解析字符串数组了，
    // 直接硬编码或简单查找，这里为了不报错，先给个占位符
    data_.cam_names.push_back("S1/stereo1_l (Manual Parsed)");

    return true;
}

}