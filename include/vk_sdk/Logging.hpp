#ifndef VK_SDK_LOGGING_HPP
#define VK_SDK_LOGGING_HPP

#include <functional>
#include <string_view>

#include "Preamble.hpp"

namespace vkc {
    /// Severity level of a log.
    enum class LogLevel {
        TRACE,
        DEBUG,
        INFO,
        WARN,
        ERROR,
    };

    /// Type of the callback used for handling logging.
    using LoggingCallback = std::function<void(LogLevel, std::string_view)>;

    /// Install a callback that handles logs produced by `vk_sdk`.
    ///
    /// If a callback has already been installed, calling this method replaces that callback.
    ///
    /// By default, a simple callback that write to the standard output is pre-installed.
    VK_SDK_API void installLoggingCallback(LoggingCallback callback);

    /// Return a human-readable string for the given log level.
    VK_SDK_API std::string_view convertLogLevelToString(LogLevel level);

    /// Log a message.
    VK_SDK_API void log(LogLevel level, std::string_view message);
}

#endif