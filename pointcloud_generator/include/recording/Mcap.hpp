#ifndef VKC_TOOLS_MCAP_HPP
#define VKC_TOOLS_MCAP_HPP

#include <vector>
#include <mcap/writer.hpp>
#include <mcap/reader.hpp>
#include <vk_sdk/Sdk.hpp>

namespace vkc {

    // Abstract class for an MCAP data source.
    class McapSource: public DataSource {
    public:
        // Get all possible topics in the source.
        virtual const std::vector<std::string>& getTopics() = 0;

        // Constructor to create a MCAP data source from file.
        static std::unique_ptr<McapSource> create(std::string_view filePath, mcap::ReadMessageOptions options);

        // Constructor to create a MCAP data source from file with playback.
        static std::unique_ptr<McapSource> create(std::string_view filePath, mcap::ReadMessageOptions options, double playbackRate);
    };


    // Abstract class for an MCAP data sink.
    class McapSink: public DataSink {
    public:
        // Constructor to create a MCAP data sink to file.
        // Has an optional argument to specify frameskips for each topic.
        // See vkc::Writer::topicFrameskips for more information.
        static std::unique_ptr<McapSink> create(
            std::string_view filePath, mcap::McapWriterOptions options, 
            std::unordered_map<std::string, uint32_t> topicFrameskips = {}
        );
    };

}

#endif 