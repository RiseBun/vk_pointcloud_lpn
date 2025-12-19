#ifndef VKC_SDK_UTILITIES_HPP
#define VKC_SDK_UTILITIES_HPP

#include "Preamble.hpp"

#include <variant>
#include <string>

namespace vkc {
    /// Helper class to identify the correct overloaded method to call.
    template<typename T>
    class Type {
        std::monostate _;
    };

    /// Block the thread until the CTRL-C signal has been received by the process.
    ///
    /// Returns `true` if waiting happened and the CTRL-C signal was received, `false` 
    /// if no waiting happened due to error.
    VK_SDK_API bool waitForCtrlCSignal();

    struct ProductId {
        // Full product ID string in the format "projectId.batchNo.unitNo"
        std::string fullProductId;

        // Parsed data
        std::string projectId;
        std::string batchNo;
        std::string unitNo;
    };

    /// Parse a product ID string into a ProductId struct.
    ///
    /// The input string is expected to be in the format "projectId.batchNo.unitNo".
    /// If the format is invalid, an exception is thrown.
    VK_SDK_API ProductId parseProductId(const std::string& productIdStr);
}

#endif