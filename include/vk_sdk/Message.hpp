#ifndef VKC_SDK_MESSAGE_HPP
#define VKC_SDK_MESSAGE_HPP

#include <cstdint>
#include <optional>

namespace vkc {

    /// Metadata of a message.
    struct Metadata {
        std::optional<uint64_t> sequenceNumber = std::nullopt; //< Sequence number of message in its queue.
        std::optional<uint64_t> publishTime = std::nullopt;    //< Time instant, in miscroseconds, when the message is published.
    };

    /// Message containing a payload and its metadata.
    template<typename T>
    struct Message {
        Message() = default;
        Message(T payload) : payload(std::move(payload)) {}
        Message(T payload, Metadata metadata) : metadata(metadata), payload(std::move(payload)) {}
        
        Metadata metadata;         //< Metadata associated with the message.
        T payload;                 //< Payload of the message (i.e. the actual data being sent).
    };

}

#endif