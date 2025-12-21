#include "trajectory/CalibLoader.hpp"
#include <fstream>
#include <iostream>

namespace trajectory {

bool CalibLoader::load(const std::string& path) {
    std::ifstream is(path);
    if (!is.is_open()) {
        std::cerr << "[Error] Cannot open calibration file: " << path << std::endl;
        return false;
    }
    
    try {
        cereal::JSONInputArchive archive(is);
        
        // 核心修正：
        // 我们的 JSON 根节点是 { "value0": { ... } }
        // 所以这里显式告诉 Cereal 读取 "value0" 节点，并将其映射到 data_ 结构体
        archive(cereal::make_nvp("value0", data_));
        
        return true;
    } catch (const cereal::Exception& e) {
        std::cerr << "[Error] JSON Parsing failed: " << e.what() << std::endl;
        return false;
    }
}

}
