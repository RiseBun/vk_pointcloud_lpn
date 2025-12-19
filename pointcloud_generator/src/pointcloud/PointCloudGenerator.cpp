#include "recording/Mcap.hpp"
#include <vk_sdk/Sdk.hpp>
#include <vk_sdk/capnp/pointcloud.capnp.h>
#include <vk_sdk/Utilities.hpp>
#include <vk_sdk/Receivers.hpp>
#include <vk_sdk/VisualKit.hpp>

#include <vk_sdk/DisparityToPointCloud.hpp>

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

#include "cereal/archives/json.hpp"
#include "cereal/types/map.hpp"
#include "cereal/types/memory.hpp"
#include "cereal/types/string.hpp"
#include "cereal/types/tuple.hpp"
#include "cereal/types/utility.hpp"
#include "cereal/types/vector.hpp"

#include <spdlog_assert.h>

struct ProgramArgs
{
    std::string inputFile = "";
    std::optional<uint64_t> startTime;
    std::optional<uint64_t> endTime;
    std::string deviceVersion;
    std::string outputDir;
    // disparity -> point cloud conversion parameters
    bool store_disparity = false;
    bool store_images = false;
    bool store_clouds = false;
    bool store_cumulative = false;
    bool publish_clouds = false;
    std::string disparity_suffix = "disparity";
    std::string config_file = "";
    std::string input_cloud_topic = "";
    std::string output_topic = "";
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
    void start()
    {
        time = std::chrono::steady_clock::now();
        running = true;
    }

    void stop()
    {
        if (running)
        {
            duration += std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - time).count();
            count++;
            running = false;
        }
    }

    double getAverageDuration()
    {
        if (count == 0)
            return 0.0;
        return duration / count;
    }
    Timer &operator+=(const Timer &timer)
    {
        this->duration += timer.duration;
        this->count += timer.count;
        return *this;
    }
};

struct PointCloudGeneratorParams
{
    int disparityOffset;   // Disparity offset parameter for disparity to point cloud conversion
    float maxDepth;        // Maximum depth parameter for disparity to point cloud conversion

    // point cloud generation parameters
    bool preVoxelDownsample;                   // Whether to downsample every frame of the point cloud
    float preVoxelSize = 0.03;                 // Voxel size for pre voxel grid filtering
    bool voxelDownsample;                      // Whether to downsample the point cloud using voxel grid filtering
    float voxelSize = 0.03;                    // Voxel size for voxel grid filtering
    int sorNeighbors = 10;                     // Number of neighbors to consider for statistical outlier removal
    float sorStddev = 1.0;                     // Std dev to consider for statistical outlier removal
    bool uniformDownsample = false;            // Whether to downsample the point cloud uniformly
    int uniformDownsamplePoints = 10000;       // Number of points to downsample to
    int skipInterval = 1;                      // Number of frames to skip
    int accumulateInterval = 1;                // Number of frames to accumulate before processing
    bool radiusOutlierRemoval = false;         // Whether to apply radius outlier removal to the point cloud
    int rorPoints = 10;                        // Number of points to consider for radius outlier removal
    double rorRadius = 0.05;                   // Radius to consider for radius outlier removal
    bool remove_statistical_outlier = false;   // Whether to remove statistical outliers from the point cloud
    bool omit_left = false;                    // Whether to omit left camera
    bool omit_right = false;                   // Whether to omit right camera
    bool remove_ground = false;                // Whether to remove ground points
    bool compressed = true;                    // Whether output pcd should be compressed
    int chunkSize = 3000000;                   // Number of points to accumulate before writing to disk
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
    void serialize(Archive &ar, PointCloudGeneratorParams &m)
    {
        ar(
           cereal::make_nvp("pre_voxel_downsample", m.preVoxelDownsample),
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
        //    cereal::make_nvp("remove_ground", m.remove_ground), // not effective for stereo cloud
           cereal::make_nvp("omit_left", m.omit_left),
           cereal::make_nvp("omit_right", m.omit_right),
           cereal::make_nvp("compressed", m.compressed),
           cereal::make_nvp("chunk_size", m.chunkSize));
    }

    template <class Archive>
    void serialize(Archive &ar, ProgramArgs &m)
    {
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
           cereal::make_nvp("z_color_period", m.z_color_period));
    }
};

struct StereoOdomData
{
    vkc::Shared<vkc::Image> image;
    vkc::Shared<vkc::Disparity> disparity;
    vkc::Shared<vkc::Odometry3d> odom;
};

struct CloudOdomData
{
    vkc::Header::Reader header;
    std::optional<vkc::Shared<vkc::Odometry3d>> odom;
    std::optional<vkc::Shared<vkc::Disparity>> disparity;
    std::optional<vkc::Shared<vkc::Image>> image;
    std::shared_ptr<open3d::geometry::PointCloud> cloud;
};

struct BatchStereoOdomData
{
    std::map<std::string, CloudOdomData> data;
};

struct CloudData
{
    vkc::Shared<vkc::PointCloud> cloud;
};

using SyncMap = tbb::concurrent_unordered_map<uint64_t, StereoOdomData>;
using ProcessUnit = tbb::concurrent_bounded_queue<StereoOdomData>;
using ProcessCloudUnit = tbb::concurrent_bounded_queue<vkc::Shared<vkc::PointCloud>>;
using BatchSyncMap = tbb::concurrent_unordered_map<uint64_t, BatchStereoOdomData>;
using BatchProcessUnit = tbb::concurrent_bounded_queue<BatchStereoOdomData>;

struct Batch
{
    std::shared_ptr<BatchSyncMap> syncMap;
    std::shared_ptr<BatchProcessUnit> processUnit;
    int batch_size;
};

void onProblemCallback(const mcap::Status &status)
{
    std::cout << "Problem encountered!" << std::endl;
}

void loggingProblem(const mcap::Status &status)
{
    vkc::log(vkc::LogLevel::WARN, status.message);
}

void erase_before(std::shared_ptr<SyncMap> &syncMap, uint64_t timestamp)
{
    for (auto it = syncMap->begin(); it != syncMap->end();)
    {
        if (it->first < timestamp)
        {
            it = syncMap->unsafe_erase(it);
        }
        else
        {
            break;
        }
    }
}

void erase_before(std::shared_ptr<BatchSyncMap> &syncMap, uint64_t timestamp)
{
    for (auto it = syncMap->begin(); it != syncMap->end();)
    {
        if (it->first < timestamp)
        {
            it = syncMap->unsafe_erase(it);
        }
        else
        {
            break;
        }
    }
}

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
        : genParams(genParams), output_dir(output_dir), receiver(std::move(receiver))
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
        this->pointCloudDirPath = output_dir;
        this->config_hash = config_hash;
        this->publish_clouds = publish_clouds;
        this->store_cumulative = store_cumulative;
        this->z_color_period = z_color_period;
        this->accumulator_id = accumulator_id;
        if (store_cumulative && !std::filesystem::exists(this->pointCloudDirPath))
        {
            if (std::filesystem::create_directory(this->pointCloudDirPath))
                std::cout << "Directory created successfully.\n";
        }
        else
            std::cout << "Directory already exists.\n";

        this->publish_topic = output_topic;
    }

    std::shared_ptr<Batch> getBatch()
    {
        return this->batch;
    }

    std::shared_ptr<Timer> getTimer()
    {
        return this->timer;
    }

    void accumulatePointClouds(BatchStereoOdomData batch)
    {
        if (batch.data.size() != this->batch->batch_size)
        {
            std::cerr << "Batch size mismatch" << std::endl;
            return;
        }

        // combine clouds into one.
        std::shared_ptr<open3d::geometry::PointCloud> cloud;

        auto start = std::chrono::high_resolution_clock::now();

        if (batch.data.size() == 1)
        {
            cloud = batch.data.begin()->second.cloud;
            if (cloud->IsEmpty())
            {
                std::cerr << "No point clouds to accumulate for frame " << this->frameCount << std::endl;
                return;
            }
        }
        else if (batch.data.size() == 2)
        {
            auto it = batch.data.begin();
            auto cloud1 = it->second.cloud;
            it++;
            auto cloud2 = it->second.cloud;
            if (cloud1->IsEmpty() || cloud2->IsEmpty())
            {
                std::cerr << "No point clouds to accumulate for frame " << this->frameCount << std::endl;
                return;
            }
            if (genParams.omit_left)
            {
                cloud = cloud2;
            }
            else if (genParams.omit_right)
            {
                cloud = cloud1;
            }
            else
            {
                cloud = cloud1;
                *cloud += *cloud2;
            }
        }
        else
        {
            std::cerr << "Batch size not supported" << std::endl;
            return;
        }
        // cloud->RemoveDuplicatedPoints(); // unnecessary since most points are distinct

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "grab clouds time_taken: " << duration.count() << " points: " << cloud->points_.size() << std::endl;

        start = std::chrono::high_resolution_clock::now();
        if (genParams.uniformDownsample)
        {
            cloud = cloud->UniformDownSample(std::max(1, (int)(cloud->points_.size() / genParams.uniformDownsamplePoints)));
        }
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        // std::cout << "cloud pair voxeldown time_taken: " << duration.count() << " points: " << cloud->points_.size() << std::endl;

        if (genParams.radiusOutlierRemoval)
        {
            cloud = std::get<0>(cloud->RemoveRadiusOutliers(genParams.rorPoints, genParams.rorRadius));
        }

        if (genParams.remove_statistical_outlier)
        {
            cloud = std::get<0>(cloud->RemoveStatisticalOutliers(genParams.sorNeighbors, genParams.sorStddev));
        }
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        std::cout << "Accumulating point clouds for frame " << this->frameCount << std::endl;
        std::cout << "Number of points added in frame " << this->frameCount << ": " << cloud->points_.size() << std::endl;

        start = std::chrono::high_resolution_clock::now();
        *this->windowCloud += *cloud;
        cloud.reset();
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        // std::cout << "combine time_taken: " << duration.count() << std::endl;

        // process every accumulateInterval * skipInterval frames
        if (this->frameCount % (this->genParams.accumulateInterval * this->genParams.skipInterval) != 0)
        {
            // std::cout << "Skipping frame " << this->frameCount << std::endl;
            return;
        }

        if (this->genParams.remove_ground) {
            this->windowCloud = this->remove_ground(this->windowCloud);
        }

        std::cout << "Accumulating batch" << std::endl;

        start = std::chrono::high_resolution_clock::now();
        *this->combinedCloud += *this->windowCloud;
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);


        if (this->publish_clouds && this->receiver)
        {
            std::cout << "Publishing cloud" << std::endl;
            auto header = batch.data.begin()->second.header;
            // std::cout << "Number of points in windowCloud: " << this->windowCloud->points_.size() << std::endl;
            this->publish_cloud(this->windowCloud, header);
        }
        this->windowCloud->Clear();

        if (!this->store_cumulative)
        {
            this->combinedCloud->Clear();
        }
        else
        {
            if (this->genParams.voxelDownsample && !this->store_cumulative)
            {
                this->combinedCloud->RemoveDuplicatedPoints();
                this->combinedCloud = this->combinedCloud->VoxelDownSample(this->genParams.voxelSize);
            }
        }
        if (this->combinedCloud->points_.size() > this->genParams.chunkSize)
        {
            this->writePointCloud();
        }
    }

    vkc::Shared<vkc::PointCloud> convertToCapnpCloud(
        std::shared_ptr<open3d::geometry::PointCloud> cloud,
        vkc::Header::Reader header)
    {
        int num_points = cloud->points_.size();
        auto mmb = std::make_unique<capnp::MallocMessageBuilder>();
        vkc::PointCloud::Builder msg = mmb->getRoot<vkc::PointCloud>();

        // Set point stride of 16 bytes (4 each for XYZ and 1 each for RGB + 1 to make it aligned)
        int pointBytes = 16;
        msg.setPointStride(pointBytes);

        // The fields are X, Y, Z, R, G, B
        auto fields = msg.initFields(6);

        // X
        fields[0].setName("x");
        fields[0].setOffset(0);
        fields[0].setType(vkc::Field::NumericType::FLOAT32);

        // Y
        fields[1].setName("y");
        fields[1].setOffset(4);
        fields[1].setType(vkc::Field::NumericType::FLOAT32);

        // Z
        fields[2].setName("z");
        fields[2].setOffset(8);
        fields[2].setType(vkc::Field::NumericType::FLOAT32);

        // R
        fields[3].setName("r");
        fields[3].setOffset(12);
        fields[3].setType(vkc::Field::NumericType::UINT8);

        // G
        fields[4].setName("g");
        fields[4].setOffset(13);
        fields[4].setType(vkc::Field::NumericType::UINT8);

        // B
        fields[5].setName("b");
        fields[5].setOffset(14);
        fields[5].setType(vkc::Field::NumericType::UINT8);

        unsigned char *pcData;
        msg.initPoints(num_points * pointBytes);
        pcData = msg.getPoints().asBytes().begin();

        for (uint32_t v = 0; v < num_points; v++)
        {
            // Calculate the point coordinates while accounting for the decimation factor
            float pt_x = cloud->points_[v].x();
            float pt_y = cloud->points_[v].y();
            float pt_z = cloud->points_[v].z();

            uint8_t b = cloud->colors_[v].x() * 255;
            uint8_t g = cloud->colors_[v].y() * 255;
            uint8_t r = cloud->colors_[v].z() * 255;

            // Set the fields of the point
            *reinterpret_cast<float *>(pcData) = pt_x;
            *reinterpret_cast<float *>(pcData + 4) = pt_y;
            *reinterpret_cast<float *>(pcData + 8) = pt_z;

            *(pcData + 12) = r;
            *(pcData + 13) = g;
            *(pcData + 14) = b;

            // Shift the pointer to the next point
            pcData += pointBytes;
        }

        // Truncate point cloud buffer to only store valid points
        // auto validBytes = static_cast<capnp::uint>(num_points * pointBytes);
        // orphan.truncate(validBytes);
        msg.setHeader(header);
        msg.getHeader().setClockDomain(vkc::Header::ClockDomain::MONOTONIC);

        auto orphan = msg.disownPoints();
        msg.adoptPoints(kj::mv(orphan));

        return vkc::Shared<vkc::PointCloud>(std::move(mmb));
    }

    void publish_cloud(std::shared_ptr<open3d::geometry::PointCloud> cloud, vkc::Header::Reader header)
    {
        auto capnp_cloud = this->convertToCapnpCloud(cloud, header);

        this->receiver->handle(this->publish_topic, vkc::Message(capnp_cloud));
    }

    std::shared_ptr<open3d::geometry::PointCloud> remove_ground(std::shared_ptr<open3d::geometry::PointCloud> cloud)
    {
        Eigen::Vector4d plane_model;
        std::vector<size_t> inliers;
        std::tie(plane_model, inliers) = cloud->SegmentPlane(0.01, 10, 100);
        // Filter out points below the plane
        auto is_above_plane = [&plane_model](const Eigen::Vector3d& point) {
            return plane_model.head<3>().dot(point) + plane_model[3] >= 0;
        };

        std::vector<Eigen::Vector3d> filtered_points, filtered_colors;
        int i = 0;
        for (const auto& point : cloud->points_) {
            if (is_above_plane(point)) {
                filtered_points.push_back(point);
                filtered_colors.push_back(cloud->colors_[i]);
            } else {
                filtered_points.push_back(point);
                filtered_colors.push_back({1.0, 0, 0});
            }
            i++;
        }

        cloud->points_ = filtered_points;
        cloud->colors_ = filtered_colors;
        return cloud;
    }

    Eigen::Isometry3d eCALSe3toEigen(vkc::Se3::Reader reader)
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

    void processData()
    {
        BatchStereoOdomData unit;
        if (this->batch->processUnit->size() == 0)
        {
            std::cout << "No data to process" << std::endl;
            return;
        }
        while (this->batch->processUnit->size() > 0)
        {
            this->frameCount++;
            this->batch->processUnit->pop(unit);
            std::cout << "Combining point clouds for frame " << frameCount << std::endl;
            timer->start();
            auto start = std::chrono::high_resolution_clock::now();
            this->accumulatePointClouds(unit);
            auto stop = std::chrono::high_resolution_clock::now();
            typedef std::chrono::duration<float> fsec;
            auto duration = std::chrono::duration_cast<fsec>(stop - start);
            std::cout << "accumulate time_taken: " << duration.count() << std::endl;

            std::cout << "Done accumulating point clouds for frame " << frameCount << std::endl;
            timer->stop();
        }
    }

    std::shared_ptr<open3d::geometry::PointCloud> getCombinedCloud()
    {
        *this->combinedCloud += *this->windowCloud;
        this->windowCloud->Clear();
        return this->combinedCloud;
    }

    void writePointCloud()
    {
        auto cloud = getCombinedCloud();
        std::cout << "Cloud has " << cloud->points_.size() << " points" << std::endl;
        if (genParams.voxelDownsample)
        {
            cloud = cloud->VoxelDownSample(genParams.voxelSize);
        }

        std::string outputCloudPath = pointCloudDirPath + "/" + config_hash + "_point_cloud_combined_" + std::to_string(this->accumulator_id) + "_" + std::to_string(this->chunkCount) + ".pcd";
        std::cout << "Saving point cloud to " << outputCloudPath << std::endl;
        open3d::io::WritePointCloudOption writePointCloudOption{
            "auto",
            open3d::io::WritePointCloudOption::IsAscii::Binary,
            genParams.compressed
                ? open3d::io::WritePointCloudOption::Compressed::Compressed
                : open3d::io::WritePointCloudOption::Compressed::Uncompressed,
            true};
        open3d::io::WritePointCloud(outputCloudPath, *cloud, writePointCloudOption);
        this->chunkCount++;
        this->combinedCloud->Clear();
    }

private:
    int chunkCount = 0;
    std::shared_ptr<open3d::geometry::PointCloud> combinedCloud;
    std::shared_ptr<open3d::geometry::PointCloud> windowCloud;
    int frameCount, accumulator_id;
    std::string pointCloudDirPath;
    PointCloudGeneratorParams genParams;
    std::shared_ptr<Batch> batch;
    std::shared_ptr<Timer> timer;
    std::string output_dir, config_hash;
    uint64_t mLastModifiedCalib;

    Eigen::Affine3d body_T_left_camera;
    Eigen::Affine3d body_T_right_camera;
    Eigen::Affine3d world_T_body;
    Eigen::Affine3d world_T_firstBody;
    Eigen::Affine3d latestBody_T_firstBody;
    bool publish_clouds, store_cumulative;
    std::unique_ptr<vkc::Receiver<vkc::PointCloud>> receiver;
    std::string publish_topic;
    std::vector<std::thread> pubThreads;
    int z_color_period;
};

class PointCloudGenerator
{

public:
    PointCloudGenerator(const std::string &direction, const vkc::PointCloudParams pcParams,
                        const std::string &output_dir,
                        const std::string &hash,
                        const bool store_clouds,
                        const bool store_disparity,
                        const bool store_images,
                        const PointCloudGeneratorParams &genParams,
                        std::shared_ptr<Timer> timer,
                        std::shared_ptr<Batch> batch,
                        std::shared_ptr<PointCloudAccumulator> accumulator,
                        int z_color_period)
        : direction(direction), pcParams(pcParams), genParams(genParams),
          batch(batch), timer(timer), accumulator(accumulator)
    {
        this->syncMap = std::make_shared<SyncMap>();
        this->processUnit = std::make_shared<ProcessUnit>();
        this->processUnit->set_capacity(1);
        this->processCloudUnit = std::make_shared<ProcessCloudUnit>();
        this->processCloudUnit->set_capacity(1);
        this->geomPtr = std::make_shared<open3d::geometry::PointCloud>();
        this->frameCount = 0;
        this->store_clouds = store_clouds;
        this->store_disparity = store_disparity;
        this->store_images = store_images;
        this->publish_clouds = publish_clouds;
        this->config_hash = hash;
        this->z_color_period = z_color_period;
        if (accumulator)
        {
            std::cout << "Accumulator found for generator " << direction << std::endl;
        }

        this->pointCloudDirPath = output_dir;
        if ((store_disparity || store_images || store_clouds) && !std::filesystem::exists(this->pointCloudDirPath))
        {
            if (std::filesystem::create_directory(this->pointCloudDirPath))
                std::cout << "Directory created successfully.\n";
        }
        else
            std::cout << "Directory already exists.\n";
    }

    std::shared_ptr<SyncMap> getSyncMap()
    {
        return this->syncMap;
    }

    std::shared_ptr<ProcessUnit> getProcessUnit()
    {
        return this->processUnit;
    }

    std::shared_ptr<ProcessCloudUnit> getProcessCloudUnit()
    {
        return this->processCloudUnit;
    }

    void processData()
    {
        StereoOdomData unit;
        bool saved_disparity_json = false;
        while (true)
        {
            this->timer->start();
            this->frameCount++;
            if (!this->processUnit->try_pop(unit))
            {
                this->timer->stop();
                continue;
            }

            auto disparityReader = unit.disparity.reader();
            auto imageReader = unit.image.reader();
            auto odomReader = unit.odom.reader();

            uint64_t lastModifiedCalib = std::max(imageReader.getIntrinsic().getLastModified(),
                                                  imageReader.getExtrinsic().getLastModified());

            this->body_T_camera = this->eCALSe3toEigen(
                imageReader.getExtrinsic().getBodyFrame());
            this->latestBody_T_firstBody = this->eCALSe3toEigen(odomReader.getPose());

            if (this->genParams.omit_left && direction == "left" || this->genParams.omit_right && direction == "right")
            {
                continue;
            }

            // std::cout << "generating cloud "<< this->frameCount << std::endl;
            // extracts point cloud from individual frames
            auto cloudPtr = std::make_shared<open3d::geometry::PointCloud>();
            this->generatePointCloud(cloudPtr, unit.disparity, unit.image);
            cloudPtr = this->preprocessPointCloud(cloudPtr, this->genParams);

            if (this->store_clouds)
            {
                this->writePointCloud(cloudPtr, frameCount);
                std::cout << "Wrote cloud " << frameCount << " with " << cloudPtr->points_.size() << " points" << std::endl;
            }
            if (this->store_disparity)
            {
                if (!saved_disparity_json)
                {
                    this->writeDisparityJson(unit.disparity, frameCount);
                    saved_disparity_json = true;
                }

                this->writeDisparity(unit.disparity, frameCount);
                std::cout << "Wrote disparity " << frameCount << std::endl;
            }

            if (this->store_images)
            {
                this->writeImage(unit.image, frameCount);
            }

            CloudOdomData cloud_odom_data = {
                unit.odom.reader().getHeader(),
                unit.odom,
                unit.disparity,
                unit.image, cloudPtr};
            auto stamp = odomReader.getHeader().getStampMonotonic() + odomReader.getHeader().getClockOffset();
            auto it = this->batch->syncMap->find(stamp);
            if (it != this->batch->syncMap->end())
            {
                if (it->second.data.find(this->direction) != it->second.data.end())
                {
                    std::cerr << "Duplicate frame " << frameCount << std::endl;
                    timer->stop();
                    continue;
                }
                it->second.data.emplace(this->direction, cloud_odom_data);
                if (this->batch->batch_size == it->second.data.size())
                {
                    if (this->batch->processUnit->size() < this->batch->processUnit->capacity())
                    {
                        this->batch->processUnit->push(it->second);
                        erase_before(this->batch->syncMap, stamp);
                    }
                }
            }
            else
            {
                BatchStereoOdomData batchUnit;
                batchUnit.data.emplace(this->direction, cloud_odom_data);
                if (this->batch->batch_size == 1)
                {
                    if (this->batch->processUnit->size() < this->batch->processUnit->capacity())
                        this->batch->processUnit->push(batchUnit);
                }
                else
                {
                    this->batch->syncMap->emplace(stamp, batchUnit);
                }
            }
            this->timer->stop();
            if (this->accumulator)
            {
                this->accumulator->processData();
            }
        }
    }

    void processCloudData()
    {
        vkc::Shared<vkc::PointCloud> cloudUnit;
        while (true)
        {
            this->timer->start();
            this->frameCount++;
            if (!this->processCloudUnit->try_pop(cloudUnit)) {
                this->timer->stop();
                std::cout << "No cloud data to process" << std::endl;
                continue;
            }
            
            std::cout << "generating cloud from cloud "<< this->frameCount << std::endl;
            // extracts point cloud from individual frames
            auto cloudPtr = std::make_shared<open3d::geometry::PointCloud>();
            this->generatePointCloud(cloudPtr, cloudUnit, false);

            cloudPtr = this->preprocessPointCloud(cloudPtr, this->genParams);

            if (this->store_clouds)
            {
                this->writePointCloud(cloudPtr, frameCount);
                std::cout << "Wrote cloud " << frameCount << " with " << cloudPtr->points_.size() << " points" << std::endl;
            }
            CloudOdomData cloud_odom_data = {
                cloudUnit.reader().getHeader(), std::nullopt, std::nullopt, std::nullopt, cloudPtr};
            auto stamp = cloudUnit.reader().getHeader().getStampMonotonic() + cloudUnit.reader().getHeader().getClockOffset();
            auto it = this->batch->syncMap->find(stamp);
            BatchStereoOdomData batchUnit;
            batchUnit.data.emplace(this->direction, cloud_odom_data);
            if (this->batch->processUnit->size() < this->batch->processUnit->capacity())
                this->batch->processUnit->push(batchUnit);
            this->timer->stop();
            if (this->accumulator)
            {
                this->accumulator->processData();
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
        auto imageEncoding = imageReader.getEncoding();

        cv::Mat imageMat;

        switch (imageEncoding)
        {
        case vkc::Image::Encoding::MONO8:

            imageMat = cv::Mat(imageHeight, imageWidth, CV_8UC1,
                               const_cast<unsigned char *>(imageReader.getData().asBytes().begin()));
            cv::cvtColor(imageMat, imageMat, cv::COLOR_GRAY2RGB);
            break;

        case vkc::Image::Encoding::YUV420:

            imageMat = cv::Mat(imageHeight * 3 / 2, imageWidth, CV_8UC1,
                               const_cast<unsigned char *>(imageReader.getData().asBytes().begin()));
            cv::cvtColor(imageMat, imageMat, cv::COLOR_YUV2BGR_IYUV);
            break;

        case vkc::Image::Encoding::BGR8:

            imageMat = cv::Mat(imageHeight, imageWidth, CV_8UC3,
                               const_cast<unsigned char *>(imageReader.getData().asBytes().begin()));
            imageMat = imageMat.reshape(1, imageHeight);
            cv::cvtColor(imageMat, imageMat, cv::COLOR_BGR2RGB);
            break;

        case vkc::Image::Encoding::JPEG:

            cv::Mat mat_jpeg(1, imageSize, CV_8UC1,
                             const_cast<unsigned char *>(imageReader.getData().asBytes().begin()));
            imageMat = cv::imdecode(mat_jpeg, cv::IMREAD_COLOR);
            break;
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

        std::vector<int> fieldOffsets;
        int offsetIter = 0;
        for (const auto &field : pointCloud.reader().getFields())
        {
            auto fieldOffset = static_cast<int>(field.getOffset());
            fieldOffsets.push_back(fieldOffset);
        }

        auto pointsCount = pointCloud.reader().getPoints().asBytes().size() / pointBytes;

        for (int pt = 0; pt < pointsCount; pt++)
        {
            float pt_x = *reinterpret_cast<float *>(pcData + fieldOffsets[offsetIter++]);
            float pt_y = *reinterpret_cast<float *>(pcData + fieldOffsets[offsetIter++]);
            float pt_z = *reinterpret_cast<float *>(pcData + fieldOffsets[offsetIter++]);

            uint8_t r = *(pcData + fieldOffsets[offsetIter++]);
            uint8_t g = *(pcData + fieldOffsets[offsetIter++]);
            uint8_t b = *(pcData + fieldOffsets[offsetIter]);

            Eigen::Vector3d world_pt_body = {pt_x, pt_y, pt_z};
            if (convertToWorld)
            {
                Eigen::Vector3d point(pt_x, pt_y, pt_z);
                world_pt_body = this->latestBody_T_firstBody * this->body_T_camera * point;
            }
            cloud->points_.push_back({world_pt_body.x(), world_pt_body.y(), world_pt_body.z()});

            if (this->z_color_period > 0)
            {
                // modulo
                int z_color = std::fmod(world_pt_body.z(), static_cast<float>(this->z_color_period)) / this->z_color_period * 255;
                r = z_color;
                g = 0;
                b = 255 - z_color;
            }

            cloud->colors_.push_back(Eigen::Vector3d(static_cast<double>(r) / 255.0,
                                                        static_cast<double>(g) / 255.0,
                                                        static_cast<double>(b) / 255.0));
            pcData += pointBytes;
            offsetIter = 0;
        }
    }

    /**
     * @brief Preprocesses the point cloud by applying voxel grid filtering and statistical outlier removal
     */
    std::shared_ptr<open3d::geometry::PointCloud> preprocessPointCloud(
        std::shared_ptr<open3d::geometry::PointCloud> cloud, PointCloudGeneratorParams genParams)
    {
        if (genParams.preVoxelDownsample)
        {
            std::cout << "Pre-voxel downsampling cloud for frame " << this->frameCount
                      << " with " << std::to_string((int)cloud->points_.size()) << " points" << std::endl;
            cloud = cloud->VoxelDownSample(genParams.preVoxelSize);
            std::cout << "Post-voxel downsampling cloud for frame " << this->frameCount
                      << " with " << std::to_string((int)cloud->points_.size()) << " points" << std::endl;
        }
        return cloud;
    }

    void writePointCloud(std::shared_ptr<open3d::geometry::PointCloud> cloud, int frameCount)
    {
        std::cout << "Saving " + this->direction + " cloud for frame " << frameCount
                  << " with " << std::to_string((int)cloud->points_.size()) << " points" << std::endl;
        std::string filePath = this->pointCloudDirPath + "/" + this->config_hash + "_point_cloud_" + this->direction + "_" + std::to_string(frameCount) + ".pcd";
        open3d::io::WritePointCloud(filePath, *cloud);
    }

    void writeDisparity(vkc::Shared<vkc::Disparity> disparity, int frameCount)
    {
        // log disparity size
        auto disparityReader = disparity.reader();
        std::cout << "Disparity size: " << disparityReader.getHeight() << disparityReader.getWidth() << std::endl;
        std::cout << "Saving disparity image for frame " << frameCount << std::endl;
        auto disparityEncoding = disparityReader.getEncoding();
        if (disparityEncoding != vkc::Disparity::Encoding::DISPARITY8 &&
            disparityEncoding != vkc::Disparity::Encoding::DISPARITY16) {
            throw std::runtime_error("encoding of image not supported: " +
                static_cast<typename std::underlying_type<vkc::Disparity::Encoding>::type>(disparityEncoding));
        }
        auto cv_encoding = disparityEncoding == vkc::Disparity::Encoding::DISPARITY8 ? CV_8UC1 : CV_16UC1;
        

        cv::Mat disparityMat = cv::Mat(
            disparityReader.getHeight(),
            disparityReader.getWidth(),
            cv_encoding, const_cast<unsigned char*>(disparityReader.getData().asBytes().begin()));
        // max disparity value
        double min, max;
        cv::minMaxLoc(disparityMat, &min, &max);
        cv::normalize(disparityMat, disparityMat, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        std::string disparityPath = this->pointCloudDirPath + "/" + this->config_hash + "_disparity_" + this->direction + "_" + std::to_string(frameCount)
                                    + "_" + std::to_string(max) + ".png";

        cv::imwrite(disparityPath, disparityMat);
    }

    void writeImage(vkc::Shared<vkc::Image> image, int frameCount)
    {
        auto imageReader = image.reader();
        std::cout << "Saving image for frame " << frameCount << std::endl;
        long imageSize = imageReader.getData().size();

        uint32_t height = imageReader.getHeight();
        uint32_t width = imageReader.getWidth();
        auto encoding = imageReader.getEncoding();

        auto imageHeader = imageReader.getHeader();
        // uint64_t stamp = imageHeader.getStamp();
        std::vector<uint32_t> imageDimensions;
        auto rgbMat = cv::Mat();

        switch (encoding)
        {
        case vkc::Image::Encoding::MONO8:

            rgbMat = cv::Mat(height, width, CV_8UC1,
                             const_cast<unsigned char *>(imageReader.getData().asBytes().begin()));
            cv::cvtColor(rgbMat, rgbMat, cv::COLOR_GRAY2RGB);
            imageDimensions = {height, width, 1};
            break;

        case vkc::Image::Encoding::YUV420:

            rgbMat = cv::Mat(height * 3 / 2, width, CV_8UC1,
                             const_cast<unsigned char *>(imageReader.getData().asBytes().begin()));
            cv::cvtColor(rgbMat, rgbMat, cv::COLOR_YUV2BGR_IYUV);
            imageDimensions = {height, width, 3};
            break;

        case vkc::Image::Encoding::BGR8:

            rgbMat = cv::Mat(height, width, CV_8UC3,
                             const_cast<unsigned char *>(imageReader.getData().asBytes().begin()));
            // rgbMat = rgbMat.reshape(1, height);
            cv::cvtColor(rgbMat, rgbMat, cv::COLOR_BGR2RGB);
            imageDimensions = {height, width, 3};
            break;

        case vkc::Image::Encoding::JPEG:

            cv::Mat mat_jpeg(1, imageSize, CV_8UC1,
                             const_cast<unsigned char *>(imageReader.getData().asBytes().begin()));
            rgbMat = cv::imdecode(mat_jpeg, cv::IMREAD_COLOR);
            cv::cvtColor(rgbMat, rgbMat, cv::COLOR_BGR2RGB);
            imageDimensions = {height, width, 3};
            SPDLOG_ASSERT(rgbMat.rows == height);
            SPDLOG_ASSERT(rgbMat.cols == width);
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
        auto fx = disparityReader.getFx();
        auto fy = disparityReader.getFy();
        auto cx = disparityReader.getCx();
        auto cy = disparityReader.getCy();
        auto baseline = disparityReader.getBaseline();
        auto decimationFactor = int(disparityReader.getDecimationFactor());
        disparityCalibFile << "{\n";
        disparityCalibFile << "  \"fx\": " << fx << ",\n";
        disparityCalibFile << "  \"fy\": " << fy << ",\n";
        disparityCalibFile << "  \"cx\": " << cx << ",\n";
        disparityCalibFile << "  \"cy\": " << cy << ",\n";
        disparityCalibFile << "  \"baseline\": " << baseline << ",\n";
        disparityCalibFile << "  \"decimationFactor\": " << decimationFactor << "\n";
        disparityCalibFile << "}\n";
        disparityCalibFile.close();
    }

    inline Eigen::Isometry3d eCALSe3toEigen(vkc::Se3::Reader reader)
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

private:
    std::string direction;
    std::shared_ptr<SyncMap> syncMap;
    std::shared_ptr<ProcessUnit> processUnit;
    std::shared_ptr<ProcessCloudUnit> processCloudUnit;

    uint64_t mLastModifiedCalib;
    Eigen::Affine3d body_T_camera;
    Eigen::Affine3d world_T_body;
    Eigen::Affine3d world_T_firstBody;
    Eigen::Affine3d latestBody_T_firstBody;

    std::shared_ptr<open3d::geometry::PointCloud> geomPtr;
    int frameCount = 0;
    std::string pointCloudDirPath, config_hash;
    vkc::PointCloudParams pcParams;
    PointCloudGeneratorParams genParams;
    bool store_clouds, store_disparity, store_images, publish_clouds;
    std::shared_ptr<PointCloudAccumulator> accumulator;
    std::shared_ptr<Batch> batch;
    std::shared_ptr<Timer> timer;
    int z_color_period;
};

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
        if (it != this->syncMap->end())
        {
            it->second.disparity = disparity;
            if (it->second.image != nullptr && it->second.odom != nullptr)
            {
                this->processUnit->push(it->second);
                erase_before(this->syncMap, disparityStamp);
            }
        }
        else
        {
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
        if (it != this->syncMap->end())
        {
            it->second.image = image;
            if (it->second.disparity != nullptr && it->second.odom != nullptr)
            {
                this->processUnit->push(it->second);
                erase_before(this->syncMap, imageStamp);
            }
            std::cout << "syncmap size: " << this->syncMap->size() << std::endl;
        }
        else
        {
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
    // Constructor for OdomReceiver
    OdomReceiver(std::shared_ptr<SyncMap> syncMap, std::shared_ptr<ProcessUnit> processUnit)
        : syncMap(syncMap), processUnit(processUnit) {}

    vkc::ReceiverStatus handle(const vkc::Message<vkc::Shared<vkc::Odometry3d>> &message) override
    {
        vkc::Shared<vkc::Odometry3d> odom = message.payload;
        auto odomReader = odom.reader();
        auto odomStamp = odomReader.getHeader().getStampMonotonic() + odomReader.getHeader().getClockOffset();

        auto it = this->syncMap->find(odomStamp);
        if (it != this->syncMap->end())
        {
            it->second.odom = odom;

            if (it->second.disparity != nullptr && it->second.image != nullptr)
            {
                this->processUnit->push(it->second);
                erase_before(this->syncMap, odomStamp);
            }
        }
        else
        {
            StereoOdomData stereoOdom;
            stereoOdom.odom = odom;
            this->syncMap->emplace(odomStamp, stereoOdom);
        }

        return vkc::ReceiverStatus::Open;
    }

private:
    std::shared_ptr<SyncMap> syncMap;
    std::shared_ptr<ProcessUnit> processUnit;

    double totalDistance;
    int counts;
};

int main(int argc, char *argv[])
{
    ProgramArgs args;
    PointCloudGeneratorParams genParams;

    CLI::App app;
    app.add_option("input", args.inputFile, "Input recording file to playback.")->capture_default_str();
    app.add_option("-s,--start-time", args.startTime, "Specified start time (uint64_t) in seconds from which to play the vbag (assume start is t = 0 s).");
    app.add_option("-e,--end-time", args.endTime, "Specified end time (uint64_t) in seconds at which to stop playing the vbag (assume start is t = 0 s).");
    app.add_option("-v,--device-version", args.deviceVersion, "Device name of Visual Kit: <180, 180-P, 360>");
    app.add_option("-o,--output", args.outputDir, "Output directory for point clouds");
    app.add_flag("--store-disparity", args.store_disparity, "Store disparity images");
    app.add_flag("--store-images", args.store_images, "Store images");
    app.add_flag("--store-clouds", args.store_clouds, "Store point clouds");
    app.add_flag("--store-cumulative", args.store_cumulative, "Store cumulative point clouds");
    app.add_flag("--publish_clouds", args.publish_clouds, "Publish point clouds");
    app.add_option("--input_cloud_topic", args.input_cloud_topic, "Input topic for point clouds");
    app.add_option("--output_topic", args.output_topic, "Output topic for point clouds");
    app.add_option("--z-color-period", args.z_color_period, "Period for coloring point cloud based on z-axis");
    app.add_option("--disparity_suffix", args.disparity_suffix, "Suffix for disparity image topic");
    app.add_option("-r,--playback-rate", args.playbackRate, "Specify the playback rate (default is 1.0).");
    app.add_option("-c,--config", args.config_file, "Configuration file for the following parameters. Specified parameters will override the configuration flags.")->required();

    CLI11_PARSE(app, argc, argv);

    if (args.config_file != "")
    {

        std::ifstream ifs(args.config_file);
        if (ifs.is_open())
        {
            cereal::JSONInputArchive archive(ifs);
            archive(cereal::make_nvp("point_cloud_generator", genParams));
        }
        else
        {
            std::cout << "Could not open configuration file. Exiting..." << std::endl;
            return 0;
        }
    }

    // generate hash of the struct
    std::hash<PointCloudGeneratorParams> hasher;
    std::string hash = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    std::string full_hash = std::to_string(hasher(genParams)).substr(0, 10) + "_" + hash;
    std::shared_ptr<vkc::DataSource> source;
    auto sourceRef = std::ref(*source);

    auto visualkit = vkc::VisualKit::create(std::nullopt);
    std::vector<std::string> topics;

    if (args.inputFile != "")
    {
        std::cout << "Using bag as source" << std::endl;
        mcap::McapReader bagReader;
        uint64_t initialLogTime;
        auto openStatus = bagReader.open(args.inputFile);
        for (auto &message : bagReader.readMessages(loggingProblem))
        {
            initialLogTime = message.message.logTime;
            break;
        }

        auto summaryStatus = bagReader.readSummary(mcap::ReadSummaryMethod::NoFallbackScan, onProblemCallback);
        auto channelsMap = bagReader.channels();

        mcap::ReadMessageOptions options;
        options.readOrder = mcap::ReadMessageOptions::ReadOrder::LogTimeOrder;

        uint64_t startTime_ns, startLogTime_ns, endTime_ns, endLogTime_ns;

        if (args.startTime.has_value())
        {
            startTime_ns = static_cast<uint64_t>(args.startTime.value() * 1e9);
            startLogTime_ns = initialLogTime + startTime_ns;
            options.startTime = startLogTime_ns;
        }
        if (args.endTime.has_value())
        {
            endTime_ns = static_cast<uint64_t>(args.endTime.value() * 1e9);
            endLogTime_ns = initialLogTime + endTime_ns;
            options.endTime = endLogTime_ns;
        }

        // auto mcap_source =
        std::shared_ptr<vkc::McapSource> mcap_shared_ptr = vkc::McapSource::create(args.inputFile, options, args.playbackRate);

        source = std::static_pointer_cast<vkc::DataSource>(mcap_shared_ptr);
        sourceRef = std::ref(*source);

        // Iterate through channel map to get topic names
        for (const auto &channel : channelsMap)
        {
            auto topic = channel.second->topic;
            topics.push_back(topic);
        }
    }
    else
    {
        std::cout << "No input file specified. Using Ecal as source" << std::endl;
        sourceRef = std::ref(visualkit->source());
    }
    std::vector<std::pair<std::string, std::string>> imageAndDisparityTopics;
    std::pair<std::string, std::string> leftStereoPairS0;
    std::pair<std::string, std::string> rightStereoPairS0;
    std::pair<std::string, std::string> leftStereoPairS1;
    std::pair<std::string, std::string> rightStereoPairS1;
    std::string S0OdomTopic, S1OdomTopic;

    leftStereoPairS0.first = "S0/stereo1_l";
    rightStereoPairS0.first = "S0/stereo2_r";
    leftStereoPairS0.second = "S0/stereo1_l/" + args.disparity_suffix;
    rightStereoPairS0.second = "S0/stereo2_r/" + args.disparity_suffix;
    leftStereoPairS1.first = "S1/stereo1_l";
    rightStereoPairS1.first = "S1/stereo2_r";
    leftStereoPairS1.second = "S1/stereo1_l/" + args.disparity_suffix;
    rightStereoPairS1.second = "S1/stereo2_r/" + args.disparity_suffix;
    S0OdomTopic = "S0/vio_odom";
    S1OdomTopic = "S1/vio_odom";

    if (visualkit == nullptr)
    {
        std::cout << "Failed to create VisualKit connection." << std::endl;
        return -1;
    }

    vkc::PointCloudParams pcParams;
    pcParams.disparityOffset = genParams.disparityOffset;
    pcParams.maxDepth = genParams.maxDepth;


    std::unordered_map<std::string, std::shared_ptr<Batch>> batches;
    std::map<std::string, std::shared_ptr<Timer>> accumulator_timers;
    std::map<std::string, std::shared_ptr<PointCloudAccumulator>> accumulators;

    if (args.input_cloud_topic != "")
    {
        // create single accumulator for input cloud.
        auto pointCloudReceiver1 = visualkit->sink().obtain(args.output_topic, vkc::Type<vkc::PointCloud>());
        auto accumulator = std::make_shared<PointCloudAccumulator>(
            genParams, args.publish_clouds,
            args.store_cumulative, args.outputDir, hash,
            std::move(pointCloudReceiver1), args.output_topic, 0);
        auto timer = accumulator->getTimer();
        auto batch = accumulator->getBatch();
        batch->batch_size = 1;
        batches[args.input_cloud_topic] = batch;
        accumulator_timers[args.input_cloud_topic] = timer;
        accumulators[args.input_cloud_topic] = accumulator;
    }
    else
    {
        // Populate according to device version
        auto pointCloudReceiver1 = visualkit->sink().obtain(args.output_topic, vkc::Type<vkc::PointCloud>());
        if (args.deviceVersion == "180-P" || args.deviceVersion == "360")
        {
            imageAndDisparityTopics.push_back(leftStereoPairS1);
            imageAndDisparityTopics.push_back(rightStereoPairS1);

            auto accumulator = std::make_shared<PointCloudAccumulator>(
                genParams, args.publish_clouds,
                args.store_cumulative, args.outputDir, hash,
                std::move(pointCloudReceiver1), args.output_topic, 1);
            auto timer = accumulator->getTimer();
            auto batch = accumulator->getBatch();
            if (genParams.omit_left || genParams.omit_right)
                batch->batch_size = 1;
            else
                batch->batch_size = 2;
            if (!genParams.omit_left)
            {
                batches[leftStereoPairS1.first] = batch;
                accumulator_timers[leftStereoPairS1.first] = timer;
                if (!genParams.omit_right)
                {
                    batches[rightStereoPairS1.first] = batch;
                }
                accumulators[leftStereoPairS1.first] = accumulator;
            }
            else
            { // assume not omit right
                batches[rightStereoPairS1.first] = batch;
                accumulators[rightStereoPairS1.first] = accumulator;
            }
        }

        auto pointCloudReceiver2 = visualkit->sink().obtain(args.output_topic, vkc::Type<vkc::PointCloud>());
        if (args.deviceVersion != "180-P")
        {
            imageAndDisparityTopics.push_back(leftStereoPairS0);
            imageAndDisparityTopics.push_back(rightStereoPairS0);
            auto accumulator = std::make_shared<PointCloudAccumulator>(
                genParams, args.publish_clouds,
                args.store_cumulative,
                args.outputDir, hash,
                std::move(pointCloudReceiver2), args.output_topic, 0);
            auto timer = accumulator->getTimer();
            auto batch = accumulator->getBatch();
            if (genParams.omit_left || genParams.omit_right)
                batch->batch_size = 1;
            else
                batch->batch_size = 2;
            if (!genParams.omit_left)
            {
                batches[leftStereoPairS0.first] = batch;
                accumulator_timers[leftStereoPairS0.first] = timer;
                if (!genParams.omit_right)
                {
                    batches[rightStereoPairS0.first] = batch;
                }
                accumulators[leftStereoPairS0.first] = accumulator;
            }
            else
            { // assume not omit right
                batches[rightStereoPairS0.first] = batch;
                accumulators[rightStereoPairS0.first] = accumulator;
            }
        }
    }

    std::string direction; // either left or right
    std::map<std::string, std::shared_ptr<Timer>> generator_timers;
    
    if (args.input_cloud_topic != "") {
        direction = "cloud";
        //create single generator for pointcloud
        generator_timers[direction] = std::make_shared<Timer>();

        std::shared_ptr<PointCloudAccumulator> accumulator;
        if (accumulators.find(args.input_cloud_topic) != accumulators.end())
        {
            accumulator = accumulators[args.input_cloud_topic];
        }
        else
        {
            std::cout << "Accumulator not found for cloud" << std::endl;
        }
        auto generator = std::make_shared<PointCloudGenerator>(
            direction, pcParams, args.outputDir, hash,
            args.store_clouds, args.store_disparity, args.store_images,
            genParams, generator_timers[direction],
            batches[args.input_cloud_topic], accumulator,
            args.z_color_period);

        std::shared_ptr<SyncMap> syncMap = generator->getSyncMap();

        auto cloudReceiver = std::make_unique<PointCloudReceiver>(generator, generator->getProcessCloudUnit());
        sourceRef.get().install(args.input_cloud_topic, std::move(cloudReceiver));
        if (args.inputFile != "")
        {
            // publish input data for viewing
            vkc::connectVisualKitReceiver(args.input_cloud_topic, sourceRef.get(), visualkit->sink());
        }
        std::thread t([generator]()
        { generator->processCloudData(); });
        t.detach();
    } else {
        for (const auto &topics : imageAndDisparityTopics)
        {
            std::cout << "Creating generator for " + topics.first << std::endl;
            if (topics.first == "S0/stereo1_l" || topics.first == "S1/stereo1_l")
                direction = "left";
            else if (topics.first == "S0/stereo2_r" || topics.first == "S1/stereo2_r")
                direction = "right";

            std::string odomTopic = topics.first.find("S0") != std::string::npos
                                        ? S0OdomTopic
                                        : S1OdomTopic;

            if (genParams.omit_left && direction == "left" || genParams.omit_right && direction == "right")
            {
                continue;
            }

            generator_timers[direction] = std::make_shared<Timer>();

            auto batch = batches[topics.first];
            if (!batch || !batch->syncMap)
            {
                throw std::runtime_error("batch syncmap undefined for direction " + direction);
            }

            std::shared_ptr<PointCloudAccumulator> accumulator;
            if (accumulators.find(topics.first) != accumulators.end())
            {
                accumulator = accumulators[topics.first];
            }
            else
            {
                std::cout << "Accumulator not found for " + direction + " stereo pair" << std::endl;
            }
            auto generator = std::make_shared<PointCloudGenerator>(
                direction, pcParams, args.outputDir, hash,
                args.store_clouds, args.store_disparity, args.store_images,
                genParams, generator_timers[direction],
                batches[topics.first], accumulator,
                args.z_color_period);

            std::shared_ptr<SyncMap> syncMap = generator->getSyncMap();
            std::shared_ptr<ProcessUnit> processUnit = generator->getProcessUnit();

            std::cout << "Installing receivers for " + direction + " stereo pair" << std::endl;

            auto imageTopic = topics.first;
            auto imageReceiver = std::make_unique<ImageReceiver>(syncMap, processUnit);

            sourceRef.get().install(imageTopic, std::move(imageReceiver));

            std::cout << "Installing receivers for " + direction + " stereo disparity" << std::endl;
            auto disparityTopic = topics.second;
            auto disparityReceiver = std::make_unique<DisparityReceiver>(syncMap, processUnit);
            sourceRef.get().install(disparityTopic, std::move(disparityReceiver));

            std::cout << "Installing receivers for odometry" << std::endl;
            auto odomReceiver = std::make_unique<OdomReceiver>(syncMap, processUnit);
            sourceRef.get().install(odomTopic, std::move(odomReceiver));
            if (args.inputFile != "")
            {
                // publish input data for viewing
                vkc::connectVisualKitReceiver(imageTopic, sourceRef.get(), visualkit->sink());
                vkc::connectVisualKitReceiver(disparityTopic, sourceRef.get(), visualkit->sink());
                vkc::connectVisualKitReceiver(odomTopic, sourceRef.get(), visualkit->sink());
            }
            std::thread t([generator]() { generator->processData(); });
            t.detach();
        }
    }

    std::cout << "Starting processing" << std::endl;
    visualkit->sink().start();
    sourceRef.get().start();
    vkc::waitForCtrlCSignal();
    sourceRef.get().stop();
    visualkit->sink().stop();
    std::cout << "Stopping processing" << std::endl;

    if (args.store_cumulative) {
        for (const auto &pair : accumulators)
        {
            pair.second->writePointCloud();
        }
    }

    // accumulate generator timers
    Timer generatorTimer = std::accumulate(generator_timers.begin(), generator_timers.end(), Timer(), [](Timer &a, std::pair<std::string, std::shared_ptr<Timer>> b)
                                           {
        a += *b.second;
        return a; });

    Timer accumulatorTimer = std::accumulate(accumulator_timers.begin(), accumulator_timers.end(), Timer(), [](Timer &a, std::pair<std::string, std::shared_ptr<Timer>> b)
                                             {
        a += *b.second;
        return a; });

    // log params and average times to csv file
    std::ofstream logFile;
    logFile.open(args.outputDir + "/log.csv", std::ios_base::app);
    // if logfile empty, add headers
    if (logFile.tellp() == 0)
    {
        logFile << "hash,input_file,start_time,end_time,device_version,output_dir,store_disparity,store_images,"
                << "store_clouds,config_file,disparity_offset,max_depth,downsample,"
                << "remove_statistical_outlier,skip_interval,accumulate_interval,"
                << "voxel_downsample,voxel_size,pre_statistical_outlier_removal,pre_sor_neighbors,pre_sor_stddev,"
                << "statistical_outlier_removal,sor_neighbors,sor_stddev,radius_outlier_removal,ror_points,ror_radius,"
                << "omit_left, omit_right,generation_avg_dur,acc_avg_dur\n";
    }

    logFile << full_hash << "," << args.inputFile << "," << args.startTime.value_or(0) << "," << args.endTime.value_or(0)
            << "," << args.deviceVersion << "," << args.outputDir << "," << args.store_disparity
            << "," << args.store_images << "," << args.store_clouds << "," << args.config_file
            << "," << genParams.disparityOffset << "," << genParams.maxDepth
            << "," << genParams.uniformDownsample << "," << genParams.uniformDownsamplePoints
            << "," << genParams.skipInterval << "," << genParams.accumulateInterval
            << "," << genParams.preVoxelDownsample << "," << genParams.preVoxelSize
            << "," << genParams.voxelDownsample << "," << genParams.voxelSize<< "," << genParams.remove_statistical_outlier
            << "," << genParams.sorNeighbors << "," << genParams.sorStddev << "," << genParams.radiusOutlierRemoval
            << "," << genParams.rorPoints << "," << genParams.rorRadius << ","
            << "," << genParams.omit_left << "," << genParams.omit_right << "," << generatorTimer.getAverageDuration()
            << "," << accumulatorTimer.getAverageDuration() << std::endl;

    // write config to config json file
    std::ofstream configFile;
    configFile.open(args.outputDir + "/" + hash + "_config.json");
    cereal::JSONOutputArchive configArchive(configFile);
    configArchive(cereal::make_nvp("point_cloud_generator", genParams));

    return 0;
}
