#ifndef VKC_SDK_VISUALKIT_HPP
#define VKC_SDK_VISUALKIT_HPP

#include <memory>
#include <string_view>
#include "vk_sdk/Data.hpp"

namespace vkc {

    /// Interface to a VisualKit unit.
    class VK_SDK_API VisualKit {
    public:
        virtual ~VisualKit() {}

        /// Returns a `DataSource` instance from VisualKit.
        ///
        /// This data source is infinite and will never be exhausted (even if there is no incoming data from the network).
        ///
        /// The same instance of `DataSource` is always returned across multiple calls to this method.
        virtual DataSource& source() = 0;

        /// Returns a `DataSink` instance from VisualKit.
        ///
        /// It is not possible to send `SystemHeartbeat` type of message to this sink.
        ///
        /// The same instance of `DataSink` is always returned across multiple calls to this method.
        virtual DataSink& sink() = 0;

        /// Construct an instance of this class.
        ///
        /// If the VisualKit's manager's address is given, the constructed interface will connect 
        /// to the manager to receive heartbeat messages. Otherwise, if `std::nullopt` is passed,
        /// no heartbeat messages will sent to any receivers.
        static std::unique_ptr<VisualKit> create(std::optional<std::string_view> manager = "127.0.0.1:80");

    };

}


#endif