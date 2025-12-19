#pragma once

#include "Preamble.hpp"
#include "vk_sdk/capnp/Shared.hpp"
#include "vk_sdk/capnp/pointcloud.capnp.h"
#include "vk_sdk/capnp/disparity.capnp.h"

namespace vkc {
    /// Parameters for converting disparity to depth.
    struct PointCloudParams {
        int disparityOffset; // Dispartity offset to deal with depth calculations; this will introduce bias
        float disparityOffsetDriver = 1; // disparity offset injected from the camera driver, to be subtracted away
        float maxDepth;
    };

    /// Convert disparity to depth.
    VK_SDK_API vkc::Shared<vkc::PointCloud> convertToPointCloud(vkc::Shared<vkc::Disparity> disparity, 
                                                                unsigned char* image, 
                                                                const PointCloudParams& pcConfig,
                                                                int skip_pixel = 1);
}
