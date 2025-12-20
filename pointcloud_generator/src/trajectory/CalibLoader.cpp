#include "trajectory/CalibLoader.hpp"
#include <fstream>
namespace trajectory {
bool CalibLoader::load(const std::string& path) {
    std::ifstream is(path);
    if (!is.is_open()) return false;
    cereal::JSONInputArchive archive(is);
    archive(data_);
    return true;
}
}