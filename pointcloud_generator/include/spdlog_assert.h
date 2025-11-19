#include <spdlog/spdlog.h>
#include <cstdlib>    // for std::abort

#define SPDLOG_ASSERT(expr)                                              \
    do {                                                                 \
        if (!(expr)) {                                                   \
            spdlog::critical("Assertion failed: {}, file: {}, line: {}", \
                             #expr, __FILE__, __LINE__);                 \
            std::abort();                                                \
        }                                                                \
    } while (0)
