#pragma once

#include "Preamble.hpp"
#include "vk_sdk/capnp/Shared.hpp"
#include "vk_sdk/capnp/disparity.capnp.h"

#include <string>

namespace vkc {
    /// Parameters for converting disparity to euclidean distance.
    struct EuclideanDistParams {
        int disparityOffset; // Dispartity offset to deal with depth calculations; this will introduce bias
        float disparityOffsetDriver = 1; // disparity offset injected from the camera driver, to be subtracted away
        float maxDepth; // Maximum distance of the pixels for filtering.
        std::string distanceEncoding; // Desired encoding of the distance image - uint16_t : "distance16"; float32: "distance32";
        bool ignoreZeroDisparity; // Set to true if do not want to visualise pixels that are too far.
    };

    /**
     * Converts a given disparity image into a Euclidean distance image.
     * 
     * @param disparity The shared capnproto message containing the disparity data and information.
     * @param disparityConfig The parameters to process the disparity image into a depth image.
     * @return The shared capnproto disparity message containing the euclidean distance data and information.
     */
    VK_SDK_API vkc::Shared<vkc::Disparity> convertToEuclideanDist(vkc::Shared<vkc::Disparity> disparity, const EuclideanDistParams& euclideanDistConfig);
}
