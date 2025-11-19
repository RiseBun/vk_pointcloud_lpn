#ifndef VKC_SDK_PREAMBLE_HPP
#define VKC_SDK_PREAMBLE_HPP

#if defined _WIN32 || defined __CYGWIN__
  #ifdef VK_SDK_EXPORT
    #ifdef __GNUC__
      #define VK_SDK_API __attribute__ ((dllexport))
    #else
      #define VK_SDK_API __declspec(dllexport)
    #endif
  #else
    #ifdef __GNUC__
      #define VK_SDK_API __attribute__ ((dllimport))
    #else
      #define VK_SDK_API __declspec(dllimport)
    #endif
  #endif
#else
  #if __GNUC__ >= 4
    #define VK_SDK_API __attribute__ ((visibility ("default")))
    #define VK_SDK_INTERNAL __attribute__ ((visibility ("hidden")))
  #else
    #define VK_SDK_API
    #define VK_SDK_INTERNAL
  #endif
#endif

#endif