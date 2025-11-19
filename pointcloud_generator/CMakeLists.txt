cmake_minimum_required(VERSION 3.16)
project(vk_pointcloud VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(IS_UBUNTU_24_04 FALSE)

#======================================================
# Detect Ubuntu version
#======================================================
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  file(READ "/etc/os-release" OS_RELEASE_CONTENTS)
  if(OS_RELEASE_CONTENTS MATCHES "Ubuntu" AND OS_RELEASE_CONTENTS MATCHES "VERSION_ID=\"24.04\"")
    set(IS_UBUNTU_24_04 TRUE)
  endif()
endif()

#======================================================
# Dependencies (via prebuild cache + system)
#======================================================
list(APPEND CMAKE_PREFIX_PATH
    "${CMAKE_SOURCE_DIR}/../cache/eigen3-install/share/eigen3/cmake"
    "${CMAKE_SOURCE_DIR}/../cache/spdlog-install/lib/cmake/spdlog"
    "${CMAKE_SOURCE_DIR}/../cache/Open3D-install/lib/cmake/Open3D"
    "${CMAKE_SOURCE_DIR}/../cache/CLI11-install/share/cmake/CLI11"
    "${CMAKE_SOURCE_DIR}/../cache/cereal-install/lib/cmake/cereal"
)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(TBB REQUIRED)
find_package(fmt REQUIRED)
find_package(Open3D REQUIRED)
find_package(spdlog REQUIRED)
find_package(CLI11 REQUIRED)
find_package(cereal REQUIRED)

#======================================================
# Library directories
#======================================================
link_directories(${CMAKE_SOURCE_DIR}/../lib)

# --- OpenCV ---
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "OpenCV not found! Please install libopencv-dev.")
endif()

message(STATUS "Eigen include dir: ${EIGEN3_INCLUDE_DIR}")

# ZSTD
set(ZSTD_INSTALLED_CMAKE_CONFIG_DIR ${CMAKE_SOURCE_DIR}/../cache/zstd-install/lib/cmake/zstd)
set(zstd_DIR ${ZSTD_INSTALLED_CMAKE_CONFIG_DIR})
find_package(zstd CONFIG REQUIRED)

# LZ4
set(LZ4_INSTALLED_CMAKE_CONFIG_DIR ${CMAKE_SOURCE_DIR}/../cache/lz4-install/lib/cmake/lz4)
set(lz4_DIR ${LZ4_INSTALLED_CMAKE_CONFIG_DIR})
find_package(lz4 CONFIG REQUIRED)

#======================================================
# Include directories
#======================================================
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/../include
    ${CMAKE_SOURCE_DIR}/../cache/mcap-include
    ${CMAKE_SOURCE_DIR}/../cache/capnp-install/include
    ${OpenCV_INCLUDE_DIRS}
)

#======================================================
# Targets
#======================================================
if(NOT MSVC)
    if(NOT IS_UBUNTU_24_04)
        # -------------------------------
        # point_cloud_gen
        # -------------------------------
        add_executable(point_cloud_gen
            src/pointcloud/PointCloudGenerator.cpp
        )
        target_link_libraries(point_cloud_gen PRIVATE
            vk_common vk_modules vk_mcap vk_sdk
            Eigen3::Eigen
            ${OpenCV_LIBS}
            TBB::tbb
            Open3D::Open3D
            cereal::cereal
            CLI11::CLI11
            spdlog::spdlog
            zstd::libzstd_static
            LZ4::lz4_static
        )
        target_include_directories(point_cloud_gen PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/../include
            ${CMAKE_SOURCE_DIR}/../cache/mcap-include
            ${CMAKE_SOURCE_DIR}/../cache/capnp-install/include
        )
        set_target_properties(point_cloud_gen PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")

        # -------------------------------
        # accum_pointcloud
        # -------------------------------
        add_executable(accum_pointcloud
            src/pointcloud/AccumPointClouds.cpp
        )
        target_link_libraries(accum_pointcloud PRIVATE
            vk_sdk
            Eigen3::Eigen
            ${OpenCV_LIBS}
            Open3D::Open3D
            CLI11::CLI11
            spdlog::spdlog
            zstd::libzstd_static
            LZ4::lz4_static
        )
        target_include_directories(accum_pointcloud PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_SOURCE_DIR}/../include
            ${CMAKE_SOURCE_DIR}/../cache/mcap-include
            ${CMAKE_SOURCE_DIR}/../cache/capnp-install/include
        )
        set_target_properties(accum_pointcloud PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")

        # -------------------------------
        # grid_mapper
        # -------------------------------
        # add_executable(grid_mapper
        #     src/pointcloud/raycast.cpp
        #     src/pointcloud/GridMapper.cpp
        # )
        # target_link_libraries(grid_mapper PRIVATE
        #     vk_common vk_modules vk_sdk
        #     Eigen3::Eigen
        #     ${OpenCV_LIBS}
        #     Open3D::Open3D
        #     TBB::tbb
        #     cereal::cereal
        #     spdlog::spdlog
        #     zstd::libzstd_static
        #     LZ4::lz4_static
        # )
        # target_include_directories(grid_mapper PRIVATE
        #     ${CMAKE_CURRENT_SOURCE_DIR}/include
        #     ${CMAKE_SOURCE_DIR}/../include
        #     ${CMAKE_SOURCE_DIR}/../cache/mcap-include
        #     ${CMAKE_SOURCE_DIR}/../cache/capnp-install/include
        # )
        # set_target_properties(grid_mapper PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")

        # -------------------------------
        # Install
        # -------------------------------
        # install(TARGETS point_cloud_gen accum_pointcloud grid_mapper COMPONENT experimental)
        install(TARGETS point_cloud_gen accum_pointcloud COMPONENT experimental)
    endif()
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "/opt/vk_pointcloud" CACHE PATH "Installation Directory" FORCE)
endif()

# 安装所有可执行程序（主程序 + 测试程序）
install(TARGETS 
    point_cloud_gen  
    accum_pointcloud              
    # grid_mapper
    EXPORT EcalOdometryMavlinkBridgeTargets
    RUNTIME DESTINATION bin       # 可执行程序安装到 /opt/vk_pointcloud/bin
    LIBRARY DESTINATION lib       # 若有共享库，安装到 /opt/vk_pointcloud/lib
    ARCHIVE DESTINATION lib       # 若有静态库，安装到 /opt/vk_pointcloud/lib
)

set_target_properties(point_cloud_gen PROPERTIES INSTALL_RPATH "$ORIGIN/../lib" )
set_target_properties(accum_pointcloud PROPERTIES INSTALL_RPATH "$ORIGIN/../lib" )
# set_target_properties(grid_mapper PROPERTIES INSTALL_RPATH "$ORIGIN/../lib" )

# set(VK_SDK_LIB_PATH "/opt/vilota/lib/libvk_sdk.so.1.4.0")
# set(VK_SDK_LINK_PATH "/opt/vilota/lib/libvk_sdk.so")
set(VK_SDK_LIB_PATH "${CMAKE_SOURCE_DIR}/../lib/libvk_sdk.so.1.4.0")
set(VK_SDK_LINK_PATH "${CMAKE_SOURCE_DIR}/../lib/libvk_sdk.so")

install(
  FILES 
  ${VK_SDK_LIB_PATH}
  ${VK_SDK_LINK_PATH}
  DESTINATION lib
)

set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/" ${CMAKE_MODULE_PATH})
include(CPackConfigDeb)