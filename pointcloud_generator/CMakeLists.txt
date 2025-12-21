cmake_minimum_required(VERSION 3.16)
project(vk_pointcloud VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(IS_UBUNTU_24_04 FALSE)

#======================================================
# 1. 检测系统版本
#======================================================
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  file(READ "/etc/os-release" OS_RELEASE_CONTENTS)
  if(OS_RELEASE_CONTENTS MATCHES "Ubuntu" AND OS_RELEASE_CONTENTS MATCHES "VERSION_ID=\"24.04\"")
    set(IS_UBUNTU_24_04 TRUE)
  endif()
endif()

#======================================================
# 2. 依赖项配置 (优先使用项目内置 cache)
#======================================================
# 修正路径：确保指向项目根目录下的 cache 文件夹
list(APPEND CMAKE_PREFIX_PATH
    "${CMAKE_SOURCE_DIR}/cache/eigen3-install/share/eigen3/cmake"
    "${CMAKE_SOURCE_DIR}/cache/spdlog-install/lib/cmake/spdlog"
    "${CMAKE_SOURCE_DIR}/cache/Open3D-install/lib/cmake/Open3D"
    "${CMAKE_SOURCE_DIR}/cache/CLI11-install/share/cmake/CLI11"
    "${CMAKE_SOURCE_DIR}/cache/cereal-install/lib/cmake/cereal"
)

# 寻找基础依赖
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(TBB REQUIRED)
find_package(fmt REQUIRED)
find_package(Open3D REQUIRED)
find_package(spdlog REQUIRED)
find_package(CLI11 REQUIRED)
find_package(cereal REQUIRED)
find_package(OpenCV REQUIRED)

# 显式寻找 eCAL 依赖（main.cpp 直接使用）
find_package(eCAL REQUIRED)

# 压缩相关依赖
set(zstd_DIR "${CMAKE_SOURCE_DIR}/cache/zstd-install/lib/cmake/zstd")
find_package(zstd CONFIG REQUIRED)
set(lz4_DIR "${CMAKE_SOURCE_DIR}/cache/lz4-install/lib/cmake/lz4")
find_package(lz4 CONFIG REQUIRED)

#======================================================
# 3. 库与头文件路径设置
#======================================================
# 设置本地库搜索目录
link_directories(${CMAKE_SOURCE_DIR}/../lib)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/../include
    ${CMAKE_SOURCE_DIR}/cache/mcap-include
    ${CMAKE_SOURCE_DIR}/cache/capnp-install/include
    ${CMAKE_SOURCE_DIR}/cache/lz4-install/include
    ${CMAKE_SOURCE_DIR}/cache/zstd-install/include
    ${OpenCV_INCLUDE_DIRS}
)

#======================================================
# 4. 定义可执行目标
#======================================================
if(NOT MSVC AND NOT IS_UBUNTU_24_04)

    # --- 4.1 原始点云生成工具 ---
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

    # --- 4.2 点云拼接工具 ---
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

    # --- 4.3 轨迹优化工具 (核心目标) ---
    # 搜集所有相关源文件
    file(GLOB TRAJECTORY_SOURCES 
        "src/trajectory/main.cpp"
        "src/trajectory/TrajectoryOptimizer.cpp"
        "src/trajectory/TimeSync.cpp"
        "src/trajectory/CalibLoader.cpp"
    )
    
    add_executable(trajectory_optimizer
        ${TRAJECTORY_SOURCES}
    )
    target_link_libraries(trajectory_optimizer PRIVATE
        vk_common vk_modules vk_mcap vk_sdk
        eCAL::core
        Eigen3::Eigen
        ${OpenCV_LIBS}
        Open3D::Open3D
        cereal::cereal
        CLI11::CLI11
        spdlog::spdlog
        zstd::libzstd_static
        LZ4::lz4_static
        TBB::tbb
    )

    # 设置 RPATH
    set_target_properties(point_cloud_gen PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
    set_target_properties(accum_pointcloud PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")
    set_target_properties(trajectory_optimizer PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")

endif()

#======================================================
# 5. 安装与打包配置
#======================================================
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "/opt/vk_pointcloud" CACHE PATH "Installation Directory" FORCE)
endif()

install(TARGETS 
    point_cloud_gen  
    accum_pointcloud
    trajectory_optimizer              
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

set(VK_SDK_LIB_PATH "${CMAKE_SOURCE_DIR}/../lib/libvk_sdk.so.1.4.0")
set(VK_SDK_LINK_PATH "${CMAKE_SOURCE_DIR}/../lib/libvk_sdk.so")

install(
  FILES ${VK_SDK_LIB_PATH} ${VK_SDK_LINK_PATH}
  DESTINATION lib
)

set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/" ${CMAKE_MODULE_PATH})
include(CPackConfigDeb)