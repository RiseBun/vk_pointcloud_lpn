#pragma once

#include "Preamble.hpp"
#include "vk_sdk/capnp/Shared.hpp"
#include "vk_sdk/capnp/disparity.capnp.h"

#include <string>

namespace vkc {
    /// Parameters for converting disparity to depth.
    struct DepthParams {
        int disparityOffset; // Dispartity offset to deal with depth calculations; this will introduce bias
        float disparityOffsetDriver = 1; // disparity offset injected from the camera driver, to be subtracted away
        float maxDepth; // Maximum depth of the pixels for filtering.
        std::string depthEncoding; // Desired encoding of the depth image - uint16_t : "depth16"; float32: "depth32";
        bool ignoreZeroDisparity; // Set to true if do not want to visualise pixels that are too far.
    };

    /**
     * Converts a given disparity image into a depth image.
     * 
     * @param disparity The shared capnproto message containing the disparity data and information.
     * @param disparityConfig The parameters to process the disparity image into a depth image.
     * @return The shared capnproto disparity message containing the depth data and information.
     */
    VK_SDK_API vkc::Shared<vkc::Disparity> convertToDepth(vkc::Shared<vkc::Disparity> disparity, const DepthParams& depthConfig);
}
