#include "recording/Mcap.hpp"
#include <vk_sdk/Sdk.hpp>
#include <vk_sdk/capnp/pointcloud.capnp.h>
#include <vk_sdk/Utilities.hpp>
#include <vk_sdk/Receivers.hpp>
#include <vk_sdk/VisualKit.hpp>
#include <fstream>
#include <sstream>
#include <vk_sdk/DisparityToPointCloud.hpp>
#include <map> 

#include <Eigen/Dense>
#include <mcap/reader.hpp>
#include <opencv2/opencv.hpp>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_unordered_map.h>
#include <open3d/Open3D.h>
#include <CLI/CLI.hpp>
#include <ecal/ecal.h>

#include <iostream>
#include <memory>
#include <csignal>
#include <filesystem>
#include <thread>
#include <chrono> 
#include <algorithm>
#include <limits>

#include "cereal/archives/json.hpp"
#include "cereal/types/map.hpp"
#include "cereal/types/memory.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/tuple.hpp"
#include "cereal/types/utility.hpp"
#include "cereal/types/vector.hpp"

#include <spdlog_assert.h>

// ===================== Part 1: 结构体定义 =====================

struct ProgramArgs
{
    std::string inputFile = "";
    std::optional<uint64_t> startTime;
    std::optional<uint64_t> endTime;
    std::string deviceVersion;
    std::string outputDir;
    bool store_disparity = false;
    bool store_images = false;
    bool store_clouds = false;
    bool store_cumulative = false;
    bool publish_clouds = false;
    std::string disparity_suffix = "disparity";
    std::string config_file = "";
    std::string input_cloud_topic = "";
    std::string output_topic = "";
    std::string poseFile = ""; 
    double manualOffset = 0.0; // [新增] 手动时间偏移参数
    int z_color_period = 0;
    double playbackRate = 1.0;
};

class Timer
{
    std::chrono::steady_clock::time_point time;
    bool running = false;
    double duration = 0.0;
    int count = 0;

public:
    void start() {
        time = std::chrono::steady_clock::now();
        running = true;
    }

    void stop() {
        if (running) {
            duration += std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - time).count();
            count++;
            running = false;
        }
    }

    double getAverageDuration() {
        if (count == 0) return 0.0;
        return duration / count;
    }
    Timer &operator+=(const Timer &timer) {
        this->duration += timer.duration;
        this->count += timer.count;
        return *this;
    }
};

struct PointCloudGeneratorParams
{
    int disparityOffset;
    float maxDepth;
    bool preVoxelDownsample;
    float preVoxelSize = 0.03;
    bool voxelDownsample;
    float voxelSize = 0.03;
    int sorNeighbors = 10;
    float sorStddev = 1.0;
    bool uniformDownsample = false;
    int uniformDownsamplePoints = 10000;
    int skipInterval = 1;
    int accumulateInterval = 1;
    bool radiusOutlierRemoval = false;
    int rorPoints = 10;
    double rorRadius = 0.05;
    bool remove_statistical_outlier = false;
    bool omit_left = false;
    bool omit_right = false;
    bool remove_ground = false;
    bool compressed = true;
    int chunkSize = 3000000;
};

template <class T>
void hash_combine(std::size_t &s, const T &v)
{
    std::hash<T> h;
    s ^= h(v) + 0x9e3779b9 + (s << 6) + (s >> 2);
}

template <>
struct std::hash<PointCloudGeneratorParams>
{
    std::size_t operator()(const PointCloudGeneratorParams &params) const
    {
        std::size_t seed = 0;
        hash_combine(seed, params.disparityOffset);
        hash_combine(seed, params.maxDepth);
        hash_combine(seed, params.preVoxelDownsample);
        hash_combine(seed, params.preVoxelSize);
        hash_combine(seed, params.voxelDownsample);
        hash_combine(seed, params.voxelSize);
        hash_combine(seed, params.sorNeighbors);
        hash_combine(seed, params.sorStddev);
        hash_combine(seed, params.skipInterval);
        hash_combine(seed, params.accumulateInterval);
        hash_combine(seed, params.radiusOutlierRemoval);
        hash_combine(seed, params.rorPoints);
        hash_combine(seed, params.rorRadius);
        hash_combine(seed, params.remove_statistical_outlier);
        hash_combine(seed, params.remove_ground);
        hash_combine(seed, params.omit_left);
        hash_combine(seed, params.omit_right);
        hash_combine(seed, params.compressed);
        return seed;
    }
};

namespace cereal
{
    template <class Archive>
    void serialize(Archive &ar, PointCloudGeneratorParams &m) {
        ar(cereal::make_nvp("pre_voxel_downsample", m.preVoxelDownsample),
           cereal::make_nvp("pre_voxel_size", m.preVoxelSize),
           cereal::make_nvp("voxel_downsample", m.voxelDownsample),
           cereal::make_nvp("voxel_size", m.voxelSize),
           cereal::make_nvp("uniform_downsample", m.uniformDownsample),
           cereal::make_nvp("uniform_downsample_points", m.uniformDownsamplePoints),
           cereal::make_nvp("sor_neighbors", m.sorNeighbors),
           cereal::make_nvp("sor_stddev", m.sorStddev),
           cereal::make_nvp("skip_interval", m.skipInterval),
           cereal::make_nvp("accumulate_interval", m.accumulateInterval),
           cereal::make_nvp("radius_outlier_removal", m.radiusOutlierRemoval),
           cereal::make_nvp("ror_points", m.rorPoints),
           cereal::make_nvp("ror_radius", m.rorRadius),
           cereal::make_nvp("disparity_offset", m.disparityOffset),
           cereal::make_nvp("max_depth", m.maxDepth),
           cereal::make_nvp("remove_statistical_outlier", m.remove_statistical_outlier),
           cereal::make_nvp("omit_left", m.omit_left),
           cereal::make_nvp("omit_right", m.omit_right),
           cereal::make_nvp("compressed", m.compressed),
           cereal::make_nvp("chunk_size", m.chunkSize));
    }

    template <class Archive>
    void serialize(Archive &ar, ProgramArgs &m) {
        ar(cereal::make_nvp("input_file", m.inputFile),
           cereal::make_nvp("start_time", m.startTime),
           cereal::make_nvp("end_time", m.endTime),
           cereal::make_nvp("device_version", m.deviceVersion),
           cereal::make_nvp("output_dir", m.outputDir),
           cereal::make_nvp("store_disparity", m.store_disparity),
           cereal::make_nvp("store_images", m.store_images),
           cereal::make_nvp("store_clouds", m.store_clouds),
           cereal::make_nvp("store_cumulative", m.store_cumulative),
           cereal::make_nvp("publish_clouds", m.publish_clouds),
           cereal::make_nvp("config_file", m.config_file),
           cereal::make_nvp("input_cloud_topic,", m.input_cloud_topic),
           cereal::make_nvp("output_topic,", m.output_topic),
           cereal::make_nvp("pose_file", m.poseFile), 
           cereal::make_nvp("z_color_period", m.z_color_period));
    }
};

struct StereoOdomData {
    vkc::Shared<vkc::Image> image;
    vkc::Shared<vkc::Disparity> disparity;
    vkc::Shared<vkc::Odometry3d> odom;
};

struct CloudOdomData {
    vkc::Header::Reader header;
    std::optional<vkc::Shared<vkc::Odometry3d>> odom;
    std::optional<vkc::Shared<vkc::Disparity>> disparity;
    std::optional<vkc::Shared<vkc::Image>> image;
    std::shared_ptr<open3d::geometry::PointCloud> cloud;
};

struct BatchStereoOdomData {
    std::map<std::string, CloudOdomData> data;
};

using SyncMap = tbb::concurrent_unordered_map<uint64_t, StereoOdomData>;
using ProcessUnit = tbb::concurrent_bounded_queue<StereoOdomData>;
using ProcessCloudUnit = tbb::concurrent_bounded_queue<vkc::Shared<vkc::PointCloud>>;
using BatchSyncMap = tbb::concurrent_unordered_map<uint64_t, BatchStereoOdomData>;
using BatchProcessUnit = tbb::concurrent_bounded_queue<BatchStereoOdomData>;

struct Batch {
    std::shared_ptr<BatchSyncMap> syncMap;
    std::shared_ptr<BatchProcessUnit> processUnit;
    int batch_size;
};

// ===================== Part 2: 辅助函数 =====================

void onProblemCallback(const mcap::Status &status) {
    std::cout << "Problem encountered!" << std::endl;
}

void loggingProblem(const mcap::Status &status) {
    vkc::log(vkc::LogLevel::WARN, status.message);
}

void erase_before(std::shared_ptr<SyncMap> &syncMap, uint64_t timestamp) {
    for (auto it = syncMap->begin(); it != syncMap->end();) {
        if (it->first < timestamp) it = syncMap->unsafe_erase(it);
        else break;
    }
}

void erase_before(std::shared_ptr<BatchSyncMap> &syncMap, uint64_t timestamp) {
    for (auto it = syncMap->begin(); it != syncMap->end();) {
        if (it->first < timestamp) it = syncMap->unsafe_erase(it);
        else break;
    }
}

static Eigen::Isometry3d eCALSe3toEigen(vkc::Se3::Reader reader)
{
    Eigen::Vector3d position = {
        reader.getPosition().getX(),
        reader.getPosition().getY(),
        reader.getPosition().getZ()};

    Eigen::Quaterniond orientation = {
        reader.getOrientation().getW(),
        reader.getOrientation().getX(),
        reader.getOrientation().getY(),
        reader.getOrientation().getZ()};

    orientation.normalize();
    Eigen::Isometry3d T;
    T.linear() = orientation.toRotationMatrix();
    T.translation() = position;
    return T;
}

// ===================== Part 3: PointCloudAccumulator =====================

class PointCloudAccumulator
{
public:
    PointCloudAccumulator(
        const PointCloudGeneratorParams &genParams,
        const bool &publish_clouds,
        const bool &store_cumulative,
        const std::string &output_dir,
        const std::string &config_hash,
        std::unique_ptr<vkc::Receiver<vkc::PointCloud>> receiver,
        const std::string &output_topic,
        const int accumulator_id)
        : genParams(genParams), 
          publish_clouds(publish_clouds),
          store_cumulative(store_cumulative),
          pointCloudDirPath(output_dir),
          config_hash(config_hash),
          receiver(std::move(receiver)),
          publish_topic(output_topic),
          accumulator_id(accumulator_id)
    {
        this->batch = std::make_shared<Batch>();
        this->batch->syncMap = std::make_shared<BatchSyncMap>();
        this->batch->processUnit = std::make_shared<BatchProcessUnit>();
        this->batch->processUnit->set_capacity(1);
        this->timer = std::make_shared<Timer>();

        this->combinedCloud = std::make_shared<open3d::geometry::PointCloud>();
        this->frameCount = 0;
        this->combinedCloud->points_.reserve(1000000);
        this->combinedCloud->colors_.reserve(1000000);

        this->windowCloud = std::make_shared<open3d::geometry::PointCloud>();
        
        if (store_cumulative && !std::filesystem::exists(this->pointCloudDirPath)) {
            if (std::filesystem::create_directory(this->pointCloudDirPath))
                std::cout << "Directory created successfully.\n";
        } else if (store_cumulative) {
            std::cout << "Directory already exists.\n";
        }
    }

    std::shared_ptr<Batch> getBatch() { return this->batch; }
    std::shared_ptr<Timer> getTimer() { return this->timer; }

    void accumulatePointClouds(BatchStereoOdomData batch)
    {
        if (batch.data.size() != this->batch->batch_size) return;

        std::shared_ptr<open3d::geometry::PointCloud> cloud;
        if (batch.data.size() == 1) {
            cloud = batch.data.begin()->second.cloud;
            if (cloud->IsEmpty()) return;
        } else if (batch.data.size() == 2) {
            auto it = batch.data.begin();
            auto cloud1 = it->second.cloud;
            it++;
            auto cloud2 = it->second.cloud;
            
            if (cloud1->IsEmpty() || cloud2->IsEmpty()) return;
            if (genParams.omit_left) cloud = cloud2;
            else if (genParams.omit_right) cloud = cloud1;
            else {
                cloud = cloud1;
                *cloud += *cloud2;
            }
        } else {
            return;
        }

        if (genParams.uniformDownsample) {
            cloud = cloud->UniformDownSample(std::max(1, (int)(cloud->points_.size() / genParams.uniformDownsamplePoints)));
        }
        
        // 注意：这里的滤波是 Accumulator 的逻辑，主要起作用的是 PointCloudGenerator 里的预处理
        if (genParams.radiusOutlierRemoval) {
            cloud = std::get<0>(cloud->RemoveRadiusOutliers(genParams.rorPoints, genParams.rorRadius));
        }
        if (genParams.remove_statistical_outlier) {
            cloud = std::get<0>(cloud->RemoveStatisticalOutliers(genParams.sorNeighbors, genParams.sorStddev));
        }

        *this->windowCloud += *cloud;
        cloud.reset();

        if (this->frameCount % (this->genParams.accumulateInterval * this->genParams.skipInterval) != 0) {
            return;
        }

        if (this->genParams.remove_ground) {
            this->windowCloud = this->remove_ground(this->windowCloud);
        }

        *this->combinedCloud += *this->windowCloud;

        if (this->publish_clouds && this->receiver) {
            auto header = batch.data.begin()->second.header;
            this->publish_cloud(this->windowCloud, header);
        }
        this->windowCloud->Clear();

        if (!this->store_cumulative) {
            this->combinedCloud->Clear();
        } else {
            if (this->genParams.voxelDownsample) {
                this->combinedCloud->RemoveDuplicatedPoints();
                this->combinedCloud = this->combinedCloud->VoxelDownSample(this->genParams.voxelSize);
            }
            if (this->combinedCloud->points_.size() > this->genParams.chunkSize) {
                this->writePointCloud();
            }
        }
    }

    void processData() { } 

    void writePointCloud() {
        static int chunk_idx = 0;
        if (this->combinedCloud->IsEmpty()) return;
        std::string filename = this->pointCloudDirPath + "/" + this->config_hash + "_accumulated_" + std::to_string(this->accumulator_id) + "_chunk_" + std::to_string(chunk_idx++) + ".pcd";
        std::cout << "Saving accumulated chunk to " << filename << std::endl;
        open3d::io::WritePointCloud(filename, *this->combinedCloud);
        this->combinedCloud->Clear();
    }

private:
    vkc::Shared<vkc::PointCloud> convertToCapnpCloud(std::shared_ptr<open3d::geometry::PointCloud> cloud, vkc::Header::Reader header) {
        int num_points = cloud->points_.size();
        auto mmb = std::make_unique<capnp::MallocMessageBuilder>();
        vkc::PointCloud::Builder msg = mmb->getRoot<vkc::PointCloud>();

        int pointBytes = 16;
        msg.setPointStride(pointBytes);
        msg.initFields(6);
        msg.initPoints(num_points * pointBytes);
        unsigned char *pcData = msg.getPoints().asBytes().begin();
        for (size_t v = 0; v < num_points; v++) {
            float pt_x = cloud->points_[v].x(), pt_y = cloud->points_[v].y(), pt_z = cloud->points_[v].z();
            uint8_t r = 0, g = 0, b = 0;
            if (v < cloud->colors_.size()) {
                b = cloud->colors_[v].x() * 255; g = cloud->colors_[v].y() * 255; r = cloud->colors_[v].z() * 255;
            }
            *reinterpret_cast<float *>(pcData) = pt_x;
            *reinterpret_cast<float *>(pcData + 4) = pt_y;
            *reinterpret_cast<float *>(pcData + 8) = pt_z;
            *(pcData + 12) = r; *(pcData + 13) = g; *(pcData + 14) = b;
            pcData += pointBytes;
        }
        msg.setHeader(header);
        msg.getHeader().setClockDomain(vkc::Header::ClockDomain::MONOTONIC);
        auto orphan = msg.disownPoints();
        msg.adoptPoints(kj::mv(orphan));
        return vkc::Shared<vkc::PointCloud>(std::move(mmb));
    }

    void publish_cloud(std::shared_ptr<open3d::geometry::PointCloud> cloud, vkc::Header::Reader header) {
        if (!this->receiver) return;
        auto capnp_cloud = this->convertToCapnpCloud(cloud, header);
        this->receiver->handle(this->publish_topic, vkc::Message(capnp_cloud));
    }

    std::shared_ptr<open3d::geometry::PointCloud> remove_ground(std::shared_ptr<open3d::geometry::PointCloud> cloud) {
        if (cloud->IsEmpty()) return cloud;
        Eigen::Vector4d plane_model;
        std::vector<size_t> inliers;
        std::tie(plane_model, inliers) = cloud->SegmentPlane(0.01, 10, 100);
        auto is_above_plane = [&plane_model](const Eigen::Vector3d& point) {
            return plane_model.head<3>().dot(point) + plane_model[3] >= 0;
        };
        std::vector<Eigen::Vector3d> filtered_points, filtered_colors;
        for (size_t i = 0; i < cloud->points_.size(); ++i) {
            if (is_above_plane(cloud->points_[i])) {
                filtered_points.push_back(cloud->points_[i]);
                if (!cloud->colors_.empty()) filtered_colors.push_back(cloud->colors_[i]);
            }
        }
        cloud->points_ = filtered_points;
        cloud->colors_ = filtered_colors;
        return cloud;
    }

    PointCloudGeneratorParams genParams;
    bool publish_clouds, store_cumulative;
    std::string pointCloudDirPath, config_hash, publish_topic;
    std::unique_ptr<vkc::Receiver<vkc::PointCloud>> receiver;
    int accumulator_id;
    std::shared_ptr<Batch> batch;
    std::shared_ptr<Timer> timer;
    std::shared_ptr<open3d::geometry::PointCloud> combinedCloud, windowCloud;
    int frameCount;
    int z_color_period = 0;
};

// ===================== Part 4: PointCloudGenerator =====================

class PointCloudGenerator
{
public:
    PointCloudGenerator(
        std::string direction,
        vkc::PointCloudParams pcParams,
        std::string outputDir,
        std::string hash,
        bool store_clouds,
        bool store_disparity,
        bool store_images,
        PointCloudGeneratorParams genParams,
        std::shared_ptr<Timer> timer,
        std::shared_ptr<Batch> batch,
        std::shared_ptr<PointCloudAccumulator> accumulator,
        int z_color_period,
        double manualOffset) 
        : direction(direction),
          pcParams(pcParams),
          pointCloudDirPath(outputDir),
          config_hash(hash),
          store_clouds(store_clouds),
          store_disparity(store_disparity),
          store_images(store_images),
          genParams(genParams),
          timer(timer),
          batch(batch),
          accumulator(accumulator),
          z_color_period(z_color_period),
          manualOffset(manualOffset)
    {
        this->syncMap = std::make_shared<SyncMap>();
        this->processUnit = std::make_shared<ProcessUnit>();
        this->processUnit->set_capacity(5);
        this->processCloudUnit = std::make_shared<ProcessCloudUnit>();
        this->processCloudUnit->set_capacity(5);
        
        this->latestBody_T_firstBody = Eigen::Affine3d::Identity();
    }

    std::shared_ptr<SyncMap> getSyncMap() { return syncMap; }
    std::shared_ptr<ProcessUnit> getProcessUnit() { return processUnit; }
    std::shared_ptr<ProcessCloudUnit> getProcessCloudUnit() { return processCloudUnit; }

    void processData()
    {
        StereoOdomData unit;
        bool saved_disparity_json = false;
        auto last_data_time = std::chrono::steady_clock::now();

        static double time_offset = 0.0;
        static bool is_offset_init = false;

        while (true)
        {
            this->timer->start();
            this->frameCount++;
            if (!this->processUnit->try_pop(unit)) {
                this->timer->stop();
                auto now = std::chrono::steady_clock::now();
                // [修改] 60秒超时
                if (this->frameCount > 10 && std::chrono::duration_cast<std::chrono::seconds>(now - last_data_time).count() > 300) {
                    std::cout << "\033[1;33m[INFO] No data received for 60 seconds. Auto-terminating...\033[0m" << std::endl;
                    std::raise(SIGINT);
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            
            last_data_time = std::chrono::steady_clock::now();

            auto disparityReader = unit.disparity.reader();
            auto imageReader = unit.image.reader();
            auto odomReader = unit.odom.reader();

            this->body_T_camera = eCALSe3toEigen(imageReader.getExtrinsic().getBodyFrame());

            // 1. 获取当前时间戳 (秒)
            uint64_t current_ts_ns = odomReader.getHeader().getStampMonotonic() + odomReader.getHeader().getClockOffset();
            double current_timestamp = static_cast<double>(current_ts_ns) * 1e-9;

            // 2. 自动计算偏移 (强制对齐)
            if (!is_offset_init && !optimized_poses_.empty()) {
                double first_pose_t = optimized_poses_.begin()->first;
                if (std::abs(current_timestamp - first_pose_t) > 10.0) {
                    time_offset = first_pose_t - current_timestamp;
                    std::cout << "\033[1;33m[AUTO-ALIGN] Detected huge time gap (" << time_offset << "s).\033[0m" << std::endl;
                }
                is_offset_init = true;
            }

            // [新增] 加上手动偏移
            double query_time = current_timestamp + time_offset + manualOffset;
            
            Eigen::Isometry3d optimized_T;
            bool found = this->getOptimizedPose(query_time, optimized_T);

            // [核心逻辑] 最小移动距离过滤 (Min Motion Filter)
            if (found) {
                Eigen::Vector3d current_pos = optimized_T.translation();
                double dist = (current_pos - last_saved_position_).norm();
                
                // 只有移动超过 5cm 才保留，或者这是第一帧
                if (last_saved_position_.x() > 90000.0 || dist > min_motion_threshold_) {
                    this->latestBody_T_firstBody = optimized_T;
                    last_saved_position_ = current_pos;
                } else {
                    // 没动，或者是初始化阶段的抖动 -> 跳过这一帧
                    this->timer->stop();
                    continue; 
                }
            } else {
                // 没找到 Pose 也跳过
                 this->timer->stop();
                 continue;
            }

            // [强力调试] 
            static int debug_count = 0;
            if (debug_count < 20) {
                std::cout << "[DEBUG Frame " << this->frameCount << "] "
                          << "MCAP: " << std::fixed << current_timestamp
                          << " | Query: " << query_time;
                if (!optimized_poses_.empty()) {
                    std::cout << " | 1st Pose: " << optimized_poses_.begin()->first;
                }
                std::cout << " | Found: " << (found ? "YES" : "NO") << std::endl;
                debug_count++;
            }

            if ((this->genParams.omit_left && direction == "left") || (this->genParams.omit_right && direction == "right")) {
                continue;
            }

            auto cloudPtr = std::make_shared<open3d::geometry::PointCloud>();
            this->generatePointCloud(cloudPtr, unit.disparity, unit.image); 
            cloudPtr = this->preprocessPointCloud(cloudPtr, this->genParams);

            if (this->store_clouds) this->writePointCloud(cloudPtr, frameCount, current_ts_ns);
            if (this->store_disparity) {
                if (!saved_disparity_json) {
                    this->writeDisparityJson(unit.disparity, frameCount);
                    saved_disparity_json = true;
                }
                this->writeDisparity(unit.disparity, frameCount);
            }
            if (this->store_images) this->writeImage(unit.image, frameCount);

            CloudOdomData cloud_odom_data = { unit.odom.reader().getHeader(), unit.odom, unit.disparity, unit.image, cloudPtr };
            auto stamp = odomReader.getHeader().getStampMonotonic() + odomReader.getHeader().getClockOffset();
            this->pushToBatch(stamp, cloud_odom_data);

            this->timer->stop();
            if (this->accumulator) this->accumulator->processData();

            if (this->frameCount % 100 == 0) {
                std::cout << "\r[Progress] Processed " << this->frameCount << " frames (" << this->direction << ")" << std::flush;
            }
        }
    }

    void processCloudData()
    {
        vkc::Shared<vkc::PointCloud> cloudUnit;
        auto last_data_time = std::chrono::steady_clock::now();
        static double time_offset = 0.0;
        static bool is_offset_init = false;

        while (true) 
        {
            this->timer->start();
            this->frameCount++;
            if (!this->processCloudUnit->try_pop(cloudUnit)) {
                this->timer->stop();
                auto now = std::chrono::steady_clock::now();
                // [修改] 60秒超时
                if (this->frameCount > 10 && std::chrono::duration_cast<std::chrono::seconds>(now - last_data_time).count() > 60) {
                    std::cout << "\033[1;33m[INFO] No data received for 60 seconds. Auto-terminating...\033[0m" << std::endl;
                    std::raise(SIGINT);
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10)); 
                continue; 
            }
            
            last_data_time = std::chrono::steady_clock::now();

            auto cloudHeader = cloudUnit.reader().getHeader();
            uint64_t cloud_ts_ns = cloudHeader.getStampMonotonic() + cloudHeader.getClockOffset();
            double current_timestamp = static_cast<double>(cloud_ts_ns) * 1e-9;

            if (!is_offset_init && !optimized_poses_.empty()) {
                double first_pose_t = optimized_poses_.begin()->first;
                if (std::abs(current_timestamp - first_pose_t) > 10.0) {
                    time_offset = first_pose_t - current_timestamp;
                    std::cout << "\033[1;33m[AUTO-ALIGN Cloud] Offset: " << time_offset << "s\033[0m" << std::endl;
                }
                is_offset_init = true;
            }

            // [新增] 加上手动偏移
            double query_time = current_timestamp + time_offset + manualOffset;

            Eigen::Isometry3d optimized_T;
            bool found = this->getOptimizedPose(query_time, optimized_T);

            // [核心逻辑] 最小移动距离过滤
            if (found) {
                Eigen::Vector3d current_pos = optimized_T.translation();
                double dist = (current_pos - last_saved_position_).norm();
                
                if (last_saved_position_.x() > 90000.0 || dist > min_motion_threshold_) {
                    this->latestBody_T_firstBody = optimized_T;
                    last_saved_position_ = current_pos;
                } else {
                    this->timer->stop();
                    continue; 
                }
            } else {
                 this->timer->stop();
                 continue;
            }

            auto cloudPtr = std::make_shared<open3d::geometry::PointCloud>();
            this->generatePointCloud(cloudPtr, cloudUnit, true); 
            cloudPtr = this->preprocessPointCloud(cloudPtr, this->genParams);

            if (this->store_clouds) this->writePointCloud(cloudPtr, frameCount, cloud_ts_ns);
            
            CloudOdomData cloud_odom_data = { cloudUnit.reader().getHeader(), std::nullopt, std::nullopt, std::nullopt, cloudPtr };
            BatchStereoOdomData batchUnit;
            batchUnit.data.emplace(this->direction, cloud_odom_data);
            if (this->batch->processUnit->size() < this->batch->processUnit->capacity())
                this->batch->processUnit->push(batchUnit);
            
            this->timer->stop();
            if (this->accumulator) this->accumulator->processData();

            if (this->frameCount % 100 == 0) {
                std::cout << "\r[Progress] Processed " << this->frameCount << " frames (Cloud)" << std::flush;
            }
        }
    }

    void loadOptimizedPoses(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "\033[1;31m[ERROR] Cannot open poses file: " << filepath << "\033[0m" << std::endl;
            return;
        }

        optimized_poses_.clear();
        std::string line;
        int count = 0;
        
        std::cout << "[INFO] Loading poses from: " << filepath << std::endl;

        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::stringstream ss(line);
            std::string ts_str; 
            double tx, ty, tz, qx, qy, qz, qw;
            if (ss >> ts_str >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
                double t = std::stod(ts_str);
                if (t > 10000000000.0) t *= 1e-9; // Auto-fix ns to sec

                Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
                T.translation() << tx, ty, tz;
                T.linear() = Eigen::Quaterniond(qw, qx, qy, qz).toRotationMatrix();

                optimized_poses_[t] = T;
                count++;
            }
        }
        
        if (optimized_poses_.empty()) {
            std::cerr << "\033[1;31m[ERROR] Loaded 0 poses! Check file format.\033[0m" << std::endl;
        } else {
            std::cout << "[INFO] Loaded " << count << " poses." << std::endl;
            if (count > 0)
                std::cout << "[INFO] Pose Time Range: " << std::fixed << optimized_poses_.begin()->first 
                          << " -> " << optimized_poses_.rbegin()->first << std::endl;
        }
    }

    bool getOptimizedPose(double timestamp, Eigen::Isometry3d& out_pose) {
        if (optimized_poses_.empty()) return false;

        auto it = optimized_poses_.lower_bound(timestamp);

        if (it == optimized_poses_.begin()) {
            out_pose = it->second;
            return true; // Always return true for first pose match
        }

        if (it == optimized_poses_.end()) {
            out_pose = std::prev(it)->second;
            return true; // Always return true for last pose match
        }

        auto it_prev = std::prev(it);
        double t1 = it_prev->first;
        double t2 = it->first;
        
        // Nearest Neighbor fallback if gap > 2s
        if (std::abs(t2 - t1) > 2.0) {
            out_pose = (std::abs(timestamp - t1) < std::abs(timestamp - t2)) ? it_prev->second : it->second;
            return true;
        }

        double alpha = (timestamp - t1) / (t2 - t1);
        Eigen::Vector3d p1 = it_prev->second.translation();
        Eigen::Vector3d p2 = it->second.translation();
        Eigen::Quaterniond q1(it_prev->second.rotation());
        Eigen::Quaterniond q2(it->second.rotation());

        out_pose = Eigen::Isometry3d::Identity();
        out_pose.translation() = p1 * (1.0 - alpha) + p2 * alpha;
        out_pose.linear() = q1.slerp(alpha, q2).toRotationMatrix();
        return true;
    }

    // ==========================================
    // [修复版] 预处理函数：真正应用去噪算法
    // ==========================================
    std::shared_ptr<open3d::geometry::PointCloud> preprocessPointCloud(
        std::shared_ptr<open3d::geometry::PointCloud> cloud, PointCloudGeneratorParams genParams)
    {
        // 1. 预体素降采样
        if (genParams.preVoxelDownsample) {
            cloud = cloud->VoxelDownSample(genParams.preVoxelSize);
        }

        // 2. [修复] 统计离群点去除 (SOR) - 去除飘在空中的稀疏噪点
        if (genParams.remove_statistical_outlier) {
            auto res = cloud->RemoveStatisticalOutliers(genParams.sorNeighbors, genParams.sorStddev);
            cloud = std::get<0>(res);
        }

        // 3. [修复] 半径离群点去除 (ROR) - 去除物体边缘的毛刺
        if (genParams.radiusOutlierRemoval) {
            auto res = cloud->RemoveRadiusOutliers(genParams.rorPoints, genParams.rorRadius);
            cloud = std::get<0>(res);
        }

        return cloud;
    }

private:
    void pushToBatch(uint64_t stamp, CloudOdomData& data) {
        auto it = this->batch->syncMap->find(stamp);
        if (it != this->batch->syncMap->end()) {
            if (it->second.data.find(this->direction) == it->second.data.end()) {
                it->second.data.emplace(this->direction, data);
                if (this->batch->batch_size == it->second.data.size()) {
                    if (this->batch->processUnit->size() < this->batch->processUnit->capacity()) {
                        this->batch->processUnit->push(it->second);
                        erase_before(this->batch->syncMap, stamp);
                    }
                }
            }
        } else {
            BatchStereoOdomData batchUnit;
            batchUnit.data.emplace(this->direction, data);
            if (this->batch->batch_size == 1) {
                if (this->batch->processUnit->size() < this->batch->processUnit->capacity())
                    this->batch->processUnit->push(batchUnit);
            } else {
                this->batch->syncMap->emplace(stamp, batchUnit);
            }
        }
    }

    void generatePointCloud(std::shared_ptr<open3d::geometry::PointCloud> cloud,
                            vkc::Shared<vkc::Disparity> disparity, vkc::Shared<vkc::Image> image)
    {
        auto imageReader = image.reader();
        long imageSize = imageReader.getData().size();
        uint32_t imageHeight = imageReader.getHeight();
        uint32_t imageWidth = imageReader.getWidth();
        cv::Mat imageMat;

        if (imageReader.getEncoding() == vkc::Image::Encoding::JPEG) {
             cv::Mat mat_jpeg(1, imageSize, CV_8UC1, const_cast<unsigned char *>(imageReader.getData().asBytes().begin()));
             imageMat = cv::imdecode(mat_jpeg, cv::IMREAD_COLOR);
        } else if (imageReader.getEncoding() == vkc::Image::Encoding::MONO8) {
            imageMat = cv::Mat(imageHeight, imageWidth, CV_8UC1, const_cast<unsigned char *>(imageReader.getData().asBytes().begin()));
            cv::cvtColor(imageMat, imageMat, cv::COLOR_GRAY2RGB);
        } else {
            imageMat = cv::Mat(imageHeight, imageWidth, CV_8UC3, const_cast<unsigned char *>(imageReader.getData().asBytes().begin()));
        }
        
        vkc::Shared<vkc::PointCloud> pointCloud = vkc::convertToPointCloud(disparity, imageMat.data, pcParams);
        generatePointCloud(cloud, pointCloud, true);
    }

    void generatePointCloud(std::shared_ptr<open3d::geometry::PointCloud> cloud,
                            vkc::Shared<vkc::PointCloud> & pointCloud,
                            bool convertToWorld = true)
    {
        unsigned char *pcData = const_cast<unsigned char *>(pointCloud.reader().getPoints().asBytes().begin());
        int pointBytes = static_cast<int>(pointCloud.reader().getPointStride());
        int pointsCount = pointCloud.reader().getPoints().asBytes().size() / pointBytes;
        
        int x_off = 0, y_off = 4, z_off = 8, r_off = 12, g_off = 13, b_off = 14; 

        for (int pt = 0; pt < pointsCount; pt++)
        {
            float pt_x = *reinterpret_cast<float *>(pcData + x_off);
            float pt_y = *reinterpret_cast<float *>(pcData + y_off);
            float pt_z = *reinterpret_cast<float *>(pcData + z_off);
            uint8_t r = *(pcData + r_off);
            uint8_t g = *(pcData + g_off);
            uint8_t b = *(pcData + b_off);

            Eigen::Vector3d world_pt_body = {pt_x, pt_y, pt_z};
            if (convertToWorld) {
                Eigen::Vector3d point_vk_raw(pt_x, pt_y, pt_z);
                Eigen::Vector3d pt_in_body = this->body_T_camera * point_vk_raw;
                world_pt_body = this->latestBody_T_firstBody * pt_in_body; 
            }
            cloud->points_.push_back(world_pt_body);
            
            if (this->z_color_period > 0) {
                 int z_color = std::fmod(world_pt_body.z(), static_cast<float>(this->z_color_period)) / this->z_color_period * 255;
                 cloud->colors_.push_back({z_color/255.0, 0.0, (255-z_color)/255.0});
            } else {
                 cloud->colors_.push_back({r/255.0, g/255.0, b/255.0});
            }
            pcData += pointBytes;
        }
    }

    void writePointCloud(std::shared_ptr<open3d::geometry::PointCloud> cloud, int frameCount, uint64_t timestamp_ns)
    {
        std::string filePath = this->pointCloudDirPath + "/" + std::to_string(timestamp_ns) + "_" + this->direction + "_" + std::to_string(frameCount) + ".pcd";
        open3d::io::WritePointCloud(filePath, *cloud);
    }
    
    void writeDisparity(vkc::Shared<vkc::Disparity> disparity, int frameCount)
    {
        auto disparityReader = disparity.reader();
        auto disparityEncoding = disparityReader.getEncoding();
        if (disparityEncoding != vkc::Disparity::Encoding::DISPARITY8 &&
            disparityEncoding != vkc::Disparity::Encoding::DISPARITY16) return;
        
        auto cv_encoding = disparityEncoding == vkc::Disparity::Encoding::DISPARITY8 ? CV_8UC1 : CV_16UC1;
        cv::Mat disparityMat = cv::Mat(
            disparityReader.getHeight(), disparityReader.getWidth(),
            cv_encoding, const_cast<unsigned char*>(disparityReader.getData().asBytes().begin()));
            
        double min, max;
        cv::minMaxLoc(disparityMat, &min, &max);
        cv::normalize(disparityMat, disparityMat, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        
        std::string disparityPath = this->pointCloudDirPath + "/" + this->config_hash + "_disparity_" + this->direction + "_" + std::to_string(frameCount) + "_" + std::to_string(max) + ".png";
        cv::imwrite(disparityPath, disparityMat);
    }

    void writeImage(vkc::Shared<vkc::Image> image, int frameCount)
    {
        auto imageReader = image.reader();
        long imageSize = imageReader.getData().size();
        uint32_t height = imageReader.getHeight();
        uint32_t width = imageReader.getWidth();
        auto encoding = imageReader.getEncoding();
        cv::Mat rgbMat;

        switch (encoding) {
        case vkc::Image::Encoding::MONO8:
            rgbMat = cv::Mat(height, width, CV_8UC1, const_cast<unsigned char *>(imageReader.getData().asBytes().begin()));
            cv::cvtColor(rgbMat, rgbMat, cv::COLOR_GRAY2RGB);
            break;
        case vkc::Image::Encoding::YUV420:
            rgbMat = cv::Mat(height * 3 / 2, width, CV_8UC1, const_cast<unsigned char *>(imageReader.getData().asBytes().begin()));
            cv::cvtColor(rgbMat, rgbMat, cv::COLOR_YUV2BGR_IYUV);
            break;
        case vkc::Image::Encoding::BGR8:
            rgbMat = cv::Mat(height, width, CV_8UC3, const_cast<unsigned char *>(imageReader.getData().asBytes().begin()));
            cv::cvtColor(rgbMat, rgbMat, cv::COLOR_BGR2RGB);
            break;
        case vkc::Image::Encoding::JPEG:
            {
                cv::Mat mat_jpeg(1, imageSize, CV_8UC1, const_cast<unsigned char *>(imageReader.getData().asBytes().begin()));
                rgbMat = cv::imdecode(mat_jpeg, cv::IMREAD_COLOR);
                cv::cvtColor(rgbMat, rgbMat, cv::COLOR_BGR2RGB);
            }
            break;
        }
        std::string imagePath = this->pointCloudDirPath + "/" + this->config_hash + "_image_" + this->direction + "_" + std::to_string(frameCount) + ".png";
        cv::imwrite(imagePath, rgbMat);
    }

    void writeDisparityJson(vkc::Shared<vkc::Disparity> disparity, int frameCount)
    {
        std::string disparityCalibPath = this->pointCloudDirPath + "/" + this->config_hash + "_disparity_calib_" + this->direction + ".json";
        std::ofstream disparityCalibFile(disparityCalibPath);
        auto disparityReader = disparity.reader();
        disparityCalibFile << "{\n";
        disparityCalibFile << "  \"fx\": " << disparityReader.getFx() << ",\n";
        disparityCalibFile << "  \"fy\": " << disparityReader.getFy() << ",\n";
        disparityCalibFile << "  \"cx\": " << disparityReader.getCx() << ",\n";
        disparityCalibFile << "  \"cy\": " << disparityReader.getCy() << ",\n";
        disparityCalibFile << "  \"baseline\": " << disparityReader.getBaseline() << ",\n";
        disparityCalibFile << "  \"decimationFactor\": " << int(disparityReader.getDecimationFactor()) << "\n";
        disparityCalibFile << "}\n";
        disparityCalibFile.close();
    }

    std::string direction;
    std::shared_ptr<SyncMap> syncMap;
    std::shared_ptr<ProcessUnit> processUnit;
    std::shared_ptr<ProcessCloudUnit> processCloudUnit;

    Eigen::Affine3d body_T_camera;
    Eigen::Affine3d latestBody_T_firstBody;

    int frameCount = 0;
    std::string pointCloudDirPath, config_hash;
    vkc::PointCloudParams pcParams;
    PointCloudGeneratorParams genParams;
    bool store_clouds, store_disparity, store_images;
    
    std::shared_ptr<PointCloudAccumulator> accumulator;
    std::shared_ptr<Batch> batch;
    std::shared_ptr<Timer> timer;
    int z_color_period;

    std::map<double, Eigen::Isometry3d> optimized_poses_;
    double first_pose_time_ = -1.0;
    
    // [新增] 最小移动距离相关变量
    Eigen::Vector3d last_saved_position_ = Eigen::Vector3d(99999.0, 99999.0, 99999.0);
    const double min_motion_threshold_ = 0.05; // 5cm
    double manualOffset;
};

// ===================== Part 5: Receivers =====================

class DisparityReceiver : public vkc::Receiver<vkc::Disparity>
{
public:
    DisparityReceiver(std::shared_ptr<SyncMap> syncMap, std::shared_ptr<ProcessUnit> processUnit)
        : syncMap(syncMap), processUnit(processUnit) {}

    vkc::ReceiverStatus handle(const vkc::Message<vkc::Shared<vkc::Disparity>> &message) override
    {
        vkc::Shared<vkc::Disparity> disparity = message.payload;
        auto disparityReader = disparity.reader();
        auto disparityStamp = disparityReader.getHeader().getStampMonotonic() + disparityReader.getHeader().getClockOffset();

        auto it = this->syncMap->find(disparityStamp);
        if (it != this->syncMap->end()) {
            it->second.disparity = disparity;
            if (it->second.image != nullptr && it->second.odom != nullptr) {
                this->processUnit->push(it->second);
                erase_before(this->syncMap, disparityStamp);
            }
        } else {
            StereoOdomData stereoOdom;
            stereoOdom.disparity = disparity;
            this->syncMap->emplace(disparityStamp, stereoOdom);
        }
        return vkc::ReceiverStatus::Open;
    }

private:
    std::shared_ptr<SyncMap> syncMap;
    std::shared_ptr<ProcessUnit> processUnit;
};

class PointCloudReceiver : public vkc::Receiver<vkc::PointCloud>
{
public:
    PointCloudReceiver(std::shared_ptr<PointCloudGenerator> generator, std::shared_ptr<ProcessCloudUnit> processCloudUnit)
        : generator(generator), processCloudUnit(processCloudUnit) {}

    vkc::ReceiverStatus handle(const vkc::Message<vkc::Shared<vkc::PointCloud>> &message) override
    {
        this->processCloudUnit->push(message.payload);
        return vkc::ReceiverStatus::Open;
    }

private:
    std::shared_ptr<PointCloudGenerator> generator;
    std::shared_ptr<ProcessCloudUnit> processCloudUnit;
};

class ImageReceiver : public vkc::Receiver<vkc::Image>
{
public:
    ImageReceiver(std::shared_ptr<SyncMap> syncMap, std::shared_ptr<ProcessUnit> processUnit)
        : syncMap(syncMap), processUnit(processUnit) {}

    vkc::ReceiverStatus handle(const vkc::Message<vkc::Shared<vkc::Image>> &message) override
    {
        vkc::Shared<vkc::Image> image = message.payload;
        auto imageReader = image.reader();
        auto imageStamp = imageReader.getHeader().getStampMonotonic() + imageReader.getHeader().getClockOffset();

        auto it = this->syncMap->find(imageStamp);
        if (it != this->syncMap->end()) {
            it->second.image = image;
            if (it->second.disparity != nullptr && it->second.odom != nullptr) {
                this->processUnit->push(it->second);
                erase_before(this->syncMap, imageStamp);
            }
        } else {
            StereoOdomData stereoOdom;
            stereoOdom.image = image;
            this->syncMap->emplace(imageStamp, stereoOdom);
        }
        return vkc::ReceiverStatus::Open;
    }

private:
    std::shared_ptr<SyncMap> syncMap;
    std::shared_ptr<ProcessUnit> processUnit;
};

class OdomReceiver : public vkc::Receiver<vkc::Odometry3d>
{
public:
    OdomReceiver(std::shared_ptr<SyncMap> syncMap, std::shared_ptr<ProcessUnit> processUnit)
        : syncMap(syncMap), processUnit(processUnit) {}

    vkc::ReceiverStatus handle(const vkc::Message<vkc::Shared<vkc::Odometry3d>> &message) override
    {
        vkc::Shared<vkc::Odometry3d> odom = message.payload;
        auto odomReader = odom.reader();
        auto odomStamp = odomReader.getHeader().getStampMonotonic() + odomReader.getHeader().getClockOffset();

        auto it = this->syncMap->find(odomStamp);
        if (it != this->syncMap->end()) {
            it->second.odom = odom;
            if (it->second.disparity != nullptr && it->second.image != nullptr) {
                this->processUnit->push(it->second);
                erase_before(this->syncMap, odomStamp);
            }
        } else {
            StereoOdomData stereoOdom;
            stereoOdom.odom = odom;
            this->syncMap->emplace(odomStamp, stereoOdom);
        }
        return vkc::ReceiverStatus::Open;
    }

private:
    std::shared_ptr<SyncMap> syncMap;
    std::shared_ptr<ProcessUnit> processUnit;
};

// ===================== Part 6: Main Function =====================

int main(int argc, char *argv[])
{
    ProgramArgs args;
    PointCloudGeneratorParams genParams;

    CLI::App app;
    app.add_option("input", args.inputFile, "Input recording file")->capture_default_str();
    app.add_option("-s,--start-time", args.startTime, "Specified start time");
    app.add_option("-e,--end-time", args.endTime, "Specified end time");
    app.add_option("-v,--device-version", args.deviceVersion, "Device name");
    app.add_option("-o,--output", args.outputDir, "Output directory");
    app.add_flag("--store-disparity", args.store_disparity, "Store disparity");
    app.add_flag("--store-images", args.store_images, "Store images");
    app.add_flag("--store-clouds", args.store_clouds, "Store point clouds");
    app.add_flag("--store-cumulative", args.store_cumulative, "Store cumulative");
    app.add_flag("--publish_clouds", args.publish_clouds, "Publish clouds");
    app.add_option("--input_cloud_topic", args.input_cloud_topic, "Input topic");
    app.add_option("--output_topic", args.output_topic, "Output topic");
    app.add_option("--z-color-period", args.z_color_period, "Z color period");
    app.add_option("--disparity_suffix", args.disparity_suffix, "Disparity suffix");
    app.add_option("-r,--playback-rate", args.playbackRate, "Playback rate");
    app.add_option("-c,--config", args.config_file, "Config file")->required();
    app.add_option("--pose-file", args.poseFile, "Path to poses.txt for global registration");
    app.add_option("--manual-offset", args.manualOffset, "Manual time offset (seconds) added to MCAP timestamp"); // [新增]

    CLI11_PARSE(app, argc, argv);

    if (args.config_file != "") {
        std::ifstream ifs(args.config_file);
        if (ifs.is_open()) {
            cereal::JSONInputArchive archive(ifs);
            archive(cereal::make_nvp("point_cloud_generator", genParams));
        }
    }

    std::hash<PointCloudGeneratorParams> hasher;
    std::string hash = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    std::string full_hash = std::to_string(hasher(genParams)).substr(0, 10) + "_" + hash;
    
    std::shared_ptr<vkc::DataSource> source;
    auto visualkit = vkc::VisualKit::create(std::nullopt);

    if (args.inputFile != "") {
        mcap::McapReader bagReader;
        uint64_t initialLogTime = 0;
        (void)bagReader.open(args.inputFile);
        for (auto &message : bagReader.readMessages(loggingProblem)) {
            initialLogTime = message.message.logTime;
            break;
        }

        mcap::ReadMessageOptions options;
        options.readOrder = mcap::ReadMessageOptions::ReadOrder::LogTimeOrder;
        if (args.startTime.has_value()) options.startTime = initialLogTime + static_cast<uint64_t>(args.startTime.value() * 1e9);
        if (args.endTime.has_value()) options.endTime = initialLogTime + static_cast<uint64_t>(args.endTime.value() * 1e9);

        std::shared_ptr<vkc::McapSource> mcap_shared_ptr = vkc::McapSource::create(args.inputFile, options, args.playbackRate);
        source = std::static_pointer_cast<vkc::DataSource>(mcap_shared_ptr);
    }
    
    vkc::DataSource* activeSource = source ? source.get() : &visualkit->source();

    std::vector<std::pair<std::string, std::string>> imageAndDisparityTopics;
    imageAndDisparityTopics.push_back({"S0/stereo1_l", "S0/stereo1_l/" + args.disparity_suffix});
    imageAndDisparityTopics.push_back({"S0/stereo2_r", "S0/stereo2_r/" + args.disparity_suffix});
    imageAndDisparityTopics.push_back({"S1/stereo1_l", "S1/stereo1_l/" + args.disparity_suffix});
    imageAndDisparityTopics.push_back({"S1/stereo2_r", "S1/stereo2_r/" + args.disparity_suffix});
    std::string S0OdomTopic = "S0/vio_odom";
    std::string S1OdomTopic = "S1/vio_odom";

    vkc::PointCloudParams pcParams;
    pcParams.disparityOffset = genParams.disparityOffset;
    pcParams.maxDepth = genParams.maxDepth;

    std::unordered_map<std::string, std::shared_ptr<Batch>> batches;
    std::map<std::string, std::shared_ptr<Timer>> accumulator_timers;
    std::map<std::string, std::shared_ptr<PointCloudAccumulator>> accumulators;

    if (args.input_cloud_topic != "") {
        auto receiver = visualkit->sink().obtain(args.output_topic, vkc::Type<vkc::PointCloud>());
        auto acc = std::make_shared<PointCloudAccumulator>(genParams, args.publish_clouds, args.store_cumulative, args.outputDir, hash, std::move(receiver), args.output_topic, 0);
        acc->getBatch()->batch_size = 1;
        batches[args.input_cloud_topic] = acc->getBatch();
        accumulators[args.input_cloud_topic] = acc;
        accumulator_timers[args.input_cloud_topic] = acc->getTimer();
    } else {
        for (auto& topicPair : imageAndDisparityTopics) {
             std::string mainTopic = topicPair.first;
             if (batches.find(mainTopic) == batches.end()) {
                 auto receiver = visualkit->sink().obtain(args.output_topic, vkc::Type<vkc::PointCloud>());
                 auto acc = std::make_shared<PointCloudAccumulator>(genParams, args.publish_clouds, args.store_cumulative, args.outputDir, hash, std::move(receiver), args.output_topic, 0);
                 acc->getBatch()->batch_size = 1; 
                 batches[mainTopic] = acc->getBatch();
                 accumulators[mainTopic] = acc;
                 accumulator_timers[mainTopic] = acc->getTimer();
             }
        }
    }

    std::map<std::string, std::shared_ptr<Timer>> generator_timers;

    if (args.input_cloud_topic != "") {
        std::string direction = "cloud";
        generator_timers[direction] = std::make_shared<Timer>();
        auto generator = std::make_shared<PointCloudGenerator>(
            direction, pcParams, args.outputDir, hash, args.store_clouds, args.store_disparity, args.store_images,
            genParams, generator_timers[direction], batches[args.input_cloud_topic], accumulators[args.input_cloud_topic], args.z_color_period, args.manualOffset);
        
        if (!args.poseFile.empty()) {
            generator->loadOptimizedPoses(args.poseFile);
        }

        auto receiver = std::make_unique<PointCloudReceiver>(generator, generator->getProcessCloudUnit());
        activeSource->install(args.input_cloud_topic, std::move(receiver));
        
        std::thread t([generator]() { generator->processCloudData(); });
        t.detach();
    } else {
        for (const auto &topics : imageAndDisparityTopics) {
            std::string direction = (topics.first.find("_l") != std::string::npos) ? "left" : "right";
            std::string odomTopic = (topics.first.find("S0") != std::string::npos) ? S0OdomTopic : S1OdomTopic;

            if ((genParams.omit_left && direction == "left") || (genParams.omit_right && direction == "right")) continue;

            generator_timers[direction] = std::make_shared<Timer>();
            auto generator = std::make_shared<PointCloudGenerator>(
                direction, pcParams, args.outputDir, hash, args.store_clouds, args.store_disparity, args.store_images,
                genParams, generator_timers[direction], batches[topics.first], accumulators[topics.first], args.z_color_period, args.manualOffset);

            if (!args.poseFile.empty()) {
                generator->loadOptimizedPoses(args.poseFile);
            }

            activeSource->install(topics.first, std::make_unique<ImageReceiver>(generator->getSyncMap(), generator->getProcessUnit()));
            activeSource->install(topics.second, std::make_unique<DisparityReceiver>(generator->getSyncMap(), generator->getProcessUnit()));
            activeSource->install(odomTopic, std::make_unique<OdomReceiver>(generator->getSyncMap(), generator->getProcessUnit()));

            std::thread t([generator]() { generator->processData(); });
            t.detach();
        }
    }

    std::cout << "Starting processing..." << std::endl;
    visualkit->sink().start();
    if (source) source->start();
    else visualkit->source().start();

    vkc::waitForCtrlCSignal();

    std::cout << "Stopping..." << std::endl;
    if (source) source->stop();
    else visualkit->source().stop();
    visualkit->sink().stop();

    if (args.store_cumulative) {
        std::cout << "Saving cumulative point clouds..." << std::endl;
        for (const auto &pair : accumulators) pair.second->writePointCloud();
    }

    return 0;
}