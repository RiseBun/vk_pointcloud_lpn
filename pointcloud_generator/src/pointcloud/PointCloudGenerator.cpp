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
#include <deque> 
#include <mutex> 
#include <optional> 

#include <Eigen/Dense>
#include <mcap/reader.hpp>
#include <opencv2/opencv.hpp>
#include <tbb/concurrent_queue.h>
#include <open3d/Open3D.h>
#include <CLI/CLI.hpp>
#include <ecal/ecal.h>

#include <iostream>
#include <iomanip> 
#include <memory>
#include <csignal>
#include <filesystem>
#include <thread>
#include <chrono> 
#include <algorithm>
#include <cstring> 

#include "cereal/archives/json.hpp"
#include "cereal/types/map.hpp"

// ===================== Part 1: 结构体定义 =====================
struct ProgramArgs {
    std::string inputFile = "";
    std::string outputDir;
    bool store_clouds = false;
    bool publish_clouds = false;
    std::string config_file = "";
    std::string output_topic = "";
    std::string poseFile = ""; 
    double manualOffset = 0.0;
    double playbackRate = 1.0;
    std::string deviceVersion;
    
    std::string input_cloud_topic = "";
    bool store_disparity = false;
    bool store_images = false;
    bool store_cumulative = false;
    int z_color_period = 0;
    std::string disparity_suffix = "disparity";
    std::optional<uint64_t> startTime;
    std::optional<uint64_t> endTime;
};

struct PointCloudGeneratorParams {
    int disparityOffset = 0;
    float maxDepth = 20.0;
    bool preVoxelDownsample = true;
    float preVoxelSize = 0.03;
    bool voxelDownsample = false;
    float voxelSize = 0.05;
    int sorNeighbors = 20;
    float sorStddev = 1.0;
    int accumulateInterval = 1; 
    bool radiusOutlierRemoval = false;
    int rorPoints = 15;
    double rorRadius = 0.05;
    bool remove_statistical_outlier = false;
    bool omit_left = false;
    bool omit_right = false;
    
    bool uniformDownsample = false;
    int uniformDownsamplePoints = 0;
    int skipInterval = 1;
    bool remove_ground = false;
    bool compressed = false;
    int chunkSize = 0;

    template <class Archive>
    void serialize(Archive &ar) {
        ar(cereal::make_nvp("pre_voxel_downsample", preVoxelDownsample),
           cereal::make_nvp("pre_voxel_size", preVoxelSize),
           cereal::make_nvp("voxel_downsample", voxelDownsample),
           cereal::make_nvp("voxel_size", voxelSize),
           cereal::make_nvp("sor_neighbors", sorNeighbors),
           cereal::make_nvp("sor_stddev", sorStddev),
           cereal::make_nvp("accumulate_interval", accumulateInterval),
           cereal::make_nvp("radius_outlier_removal", radiusOutlierRemoval),
           cereal::make_nvp("ror_points", rorPoints),
           cereal::make_nvp("ror_radius", rorRadius),
           cereal::make_nvp("disparity_offset", disparityOffset),
           cereal::make_nvp("max_depth", maxDepth),
           cereal::make_nvp("remove_statistical_outlier", remove_statistical_outlier),
           cereal::make_nvp("omit_left", omit_left),
           cereal::make_nvp("omit_right", omit_right));
    }
};

using ImageQueue = tbb::concurrent_bounded_queue<vkc::Shared<vkc::Image>>;
using DisparityQueue = tbb::concurrent_bounded_queue<vkc::Shared<vkc::Disparity>>;

// ===================== Part 2: 辅助类与函数 =====================
void loggingProblem(const mcap::Status &status) {
    vkc::log(vkc::LogLevel::WARN, status.message);
}

static Eigen::Isometry3d eCALSe3toEigen(vkc::Se3::Reader reader) {
    Eigen::Vector3d p(reader.getPosition().getX(), reader.getPosition().getY(), reader.getPosition().getZ());
    Eigen::Quaterniond q(reader.getOrientation().getW(), reader.getOrientation().getX(), reader.getOrientation().getY(), reader.getOrientation().getZ());
    return Eigen::Isometry3d(Eigen::Translation3d(p) * q.normalized());
}

class PointCloudAccumulator {
public:
    PointCloudAccumulator(const std::string &output_dir, bool publish, const std::string &topic, std::unique_ptr<vkc::Receiver<vkc::PointCloud>> rx) 
        : publish_clouds(publish), publish_topic(topic), receiver(std::move(rx)) {}

    void process(std::shared_ptr<open3d::geometry::PointCloud> cloud, vkc::Header::Reader header) {
        if (publish_clouds && receiver && !cloud->IsEmpty()) {
            auto mmb = std::make_unique<capnp::MallocMessageBuilder>();
            vkc::PointCloud::Builder msg = mmb->getRoot<vkc::PointCloud>();
            int n = cloud->points_.size();
            msg.setPointStride(16);
            msg.initPoints(n * 16);
            unsigned char *ptr = msg.getPoints().asBytes().begin();
            for (int i=0; i<n; ++i) {
                float *p = reinterpret_cast<float*>(ptr);
                p[0] = cloud->points_[i].x(); p[1] = cloud->points_[i].y(); p[2] = cloud->points_[i].z();
                if (i < cloud->colors_.size()) {
                    ptr[12] = cloud->colors_[i].z()*255; ptr[13] = cloud->colors_[i].y()*255; ptr[14] = cloud->colors_[i].x()*255; 
                }
                ptr += 16;
            }
            msg.setHeader(header);
            receiver->handle(publish_topic, vkc::Message(vkc::Shared<vkc::PointCloud>(std::move(mmb))));
        }
    }
private:
    bool publish_clouds;
    std::string publish_topic;
    std::unique_ptr<vkc::Receiver<vkc::PointCloud>> receiver;
};

// ===================== Part 3: PointCloudGenerator (3通道输入 + 强制黑白输出) =====================
class PointCloudGenerator {
public:
    PointCloudGenerator(
        vkc::PointCloudParams pcParams,
        std::string outputDir,
        bool store_clouds,
        PointCloudGeneratorParams genParams,
        std::shared_ptr<PointCloudAccumulator> accumulator,
        double manualOffset) 
        : pcParams(pcParams), pointCloudDirPath(outputDir),
          store_clouds(store_clouds), genParams(genParams), accumulator(accumulator), manualOffset(manualOffset)
    {
        leftImgQ = std::make_shared<ImageQueue>(); leftImgQ->set_capacity(200);
        leftDispQ = std::make_shared<DisparityQueue>(); leftDispQ->set_capacity(200);
        
        submap_cloud_ = std::make_shared<open3d::geometry::PointCloud>();
        if (!std::filesystem::exists(outputDir)) std::filesystem::create_directory(outputDir);
    }

    auto getLeftImgQ() { return leftImgQ; }
    auto getLeftDispQ() { return leftDispQ; }

    void loadOptimizedPoses(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) return;
        optimized_poses_.clear();
        std::string line;
        
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::replace(line.begin(), line.end(), ',', ' ');
            std::stringstream ss(line);
            std::string ts_str; 
            double tx, ty, tz, qx, qy, qz, qw;
            if (ss >> ts_str >> tx >> ty >> tz >> qx >> qy >> qz >> qw) {
                optimized_poses_[std::stod(ts_str)] = Eigen::Isometry3d(Eigen::Translation3d(tx, ty, tz) * Eigen::Quaterniond(qw, qx, qy, qz));
            }
        }
        std::cout << "[INFO] Loaded " << optimized_poses_.size() << " poses." << std::endl;
    }

    void processStereoStream() {
        std::deque<vkc::Shared<vkc::Image>> lImgBuf;
        std::deque<vkc::Shared<vkc::Disparity>> lDispBuf;

        int frameCount = 0;
        int submapCount = 0;
        std::vector<double> submapTimes;
        Eigen::Isometry3d last_keyframe_pose = Eigen::Isometry3d::Identity();
        bool first_frame = true;

        std::cout << "[INFO] Stereo processing loop started (Dictator Mode: Force Grayscale)..." << std::endl;

        while (true) {
            vkc::Shared<vkc::Image> i; while(leftImgQ->try_pop(i)) lImgBuf.push_back(i);
            vkc::Shared<vkc::Disparity> d; while(leftDispQ->try_pop(d)) lDispBuf.push_back(d);

            if (lImgBuf.empty()) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); continue; }
            
            auto currLImg = lImgBuf.front();
            uint64_t target_ns = currLImg.reader().getHeader().getStampMonotonic(); 
            
            auto currLDispOpt = findNearest(lDispBuf, target_ns, 15000000);
            if (!currLDispOpt) {
                if (lImgBuf.size() > 50) lImgBuf.pop_front(); 
                continue; 
            }
            auto currLDisp = *currLDispOpt;
            lImgBuf.pop_front(); 

            double query_time = (double)target_ns * 1e-9 + manualOffset;
            Eigen::Isometry3d T_world_body; 
            
            if (!getOptimizedPose(query_time, T_world_body)) continue;
            
            if (T_world_body.translation().norm() > 10000.0) continue;

            if (first_frame) {
                last_keyframe_pose = T_world_body;
                first_frame = false;
            } else {
                Eigen::Isometry3d delta = last_keyframe_pose.inverse() * T_world_body;
                if (delta.translation().norm() < 0.1 && Eigen::AngleAxisd(delta.rotation()).angle() < 0.05) continue; 
                last_keyframe_pose = T_world_body;
            }

            frameCount++;
            
            auto frameCloud = std::make_shared<open3d::geometry::PointCloud>();
            if (!genParams.omit_left) {
                auto c = generateCloudCam(currLDisp, currLImg);
                
                if (c && !c->IsEmpty()) {
                    Eigen::Isometry3d T_body_cam = eCALSe3toEigen(currLImg.reader().getExtrinsic().getBodyFrame());
                    Eigen::Isometry3d T_total = T_world_body * T_body_cam;
                    
                    for (auto& p : c->points_) {
                        p = T_total * p;
                    }
                    *frameCloud += *c;
                }
            }

            *submap_cloud_ += *frameCloud;
            submapCount++;
            submapTimes.push_back(query_time);

            if (submapCount >= genParams.accumulateInterval) {
                if (!submap_cloud_->IsEmpty()) {
                    auto bbox = submap_cloud_->GetAxisAlignedBoundingBox();
                    if (bbox.max_bound_.norm() > 1e6) { 
                        submap_cloud_->Clear();
                        submapCount = 0; 
                        submapTimes.clear();
                        continue;
                    }
                }

                if (!submap_cloud_->IsEmpty()) {
                    double mid_t = submapTimes[submapTimes.size()/2];
                    std::string path = pointCloudDirPath + "/" + std::to_string((uint64_t)(mid_t*1e6)) + "_submap_" + std::to_string(frameCount) + ".pcd";
                    
                    if (store_clouds) open3d::io::WritePointCloud(path, *submap_cloud_);
                    if (accumulator) accumulator->process(submap_cloud_, currLImg.reader().getHeader());

                    std::cout << "\r[Processed] Time: " << std::fixed << mid_t << " Pts: " << submap_cloud_->points_.size() << "    " << std::flush;
                }
                
                submap_cloud_->Clear();
                submapCount = 0;
                submapTimes.clear();
            }
        }
    }

private:
    template <typename T>
    std::optional<vkc::Shared<T>> findNearest(std::deque<vkc::Shared<T>>& buf, uint64_t target_ns, int64_t tol_ns) {
        std::optional<vkc::Shared<T>> best = std::nullopt;
        int64_t min_diff = tol_ns + 1;
        for (auto& item : buf) {
             uint64_t item_ts = item.reader().getHeader().getStampMonotonic();
             int64_t diff = std::abs((int64_t)target_ns - (int64_t)item_ts);
             if (diff <= tol_ns && diff < min_diff) { min_diff = diff; best = item; }
        }
        while(!buf.empty() && (int64_t)target_ns - (int64_t)buf.front().reader().getHeader().getStampMonotonic() > 500000000) buf.pop_front();
        return best;
    }

    bool getOptimizedPose(double t, Eigen::Isometry3d& out) {
        if (optimized_poses_.empty()) return false;
        auto it = optimized_poses_.lower_bound(t);
        if (it == optimized_poses_.begin()) { out = it->second; return true; }
        if (it == optimized_poses_.end()) { out = std::prev(it)->second; return true; }
        
        auto prev = std::prev(it);
        double t1 = prev->first, t2 = it->first;
        if (std::abs(t2 - t1) > 2.0) return false; 

        double alpha = (t - t1)/(t2 - t1);
        out.translation() = prev->second.translation() * (1-alpha) + it->second.translation() * alpha;
        
        Eigen::Quaterniond qprev(prev->second.rotation());
        Eigen::Quaterniond qnext(it->second.rotation());
        out.linear() = qprev.slerp(alpha, qnext).normalized().toRotationMatrix();
        return true;
    }

    std::shared_ptr<open3d::geometry::PointCloud> generateCloudCam(vkc::Shared<vkc::Disparity> disp, vkc::Shared<vkc::Image> img) {
        auto iR = img.reader();
        cv::Mat gray_3ch;
        bool decode_success = false;
        
        // 1. 解码并转为 3通道灰度 (防 SDK 崩)
        try {
            cv::Mat temp;
            if (iR.getData().size() > 0) {
                if (iR.getEncoding() == vkc::Image::Encoding::JPEG) {
                    cv::Mat raw(1, iR.getData().size(), CV_8UC1, (void*)iR.getData().begin());
                    temp = cv::imdecode(raw, cv::IMREAD_COLOR);
                } else if (iR.getEncoding() == vkc::Image::Encoding::MONO8) {
                    cv::Mat mono(iR.getHeight(), iR.getWidth(), CV_8UC1, const_cast<unsigned char *>(iR.getData().asBytes().begin()));
                    cv::cvtColor(mono, temp, cv::COLOR_GRAY2BGR);
                } else {
                    size_t expected = iR.getWidth() * iR.getHeight() * 3;
                    if (iR.getData().size() >= expected) {
                        temp = cv::Mat(iR.getHeight(), iR.getWidth(), CV_8UC3, const_cast<unsigned char *>(iR.getData().asBytes().begin()));
                    }
                }
            }
            if (!temp.empty() && temp.rows > 0 && temp.cols > 0) {
                cv::Mat gray_1ch;
                cv::cvtColor(temp, gray_1ch, cv::COLOR_BGR2GRAY);
                cv::cvtColor(gray_1ch, gray_3ch, cv::COLOR_GRAY2BGR); // 3通道灰度
                decode_success = true;
            }
        } catch (...) { decode_success = false; }

        if (!decode_success) return std::make_shared<open3d::geometry::PointCloud>();

        // 2. 调用 SDK
        auto vkCloud = vkc::convertToPointCloud(disp, gray_3ch.data, pcParams);
        
        auto out = std::make_shared<open3d::geometry::PointCloud>();
        
        // 3. 解析点云
        auto ptsReader = vkCloud.reader().getPoints();
        auto ptsData = ptsReader.asBytes();
        size_t raw_size = ptsData.size();
        const int stride = 16; 
        size_t count = raw_size / stride;
        const uint8_t* base_ptr = ptsData.begin();

        if (raw_size == 0 || base_ptr == nullptr) return out;

        for(size_t i = 0; i < count; ++i) {
            const uint8_t* p = base_ptr + i * stride;
            float x, y, z;
            std::memcpy(&x, p + 0, 4);
            std::memcpy(&y, p + 4, 4);
            std::memcpy(&z, p + 8, 4);
            
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
            if (z < 0.5f || z > pcParams.maxDepth) continue;
            if (std::abs(x) > 200.0f || std::abs(y) > 200.0f) continue;

            out->points_.push_back({(double)x, (double)y, (double)z});
            
            // [核心修正] 强制写死颜色为灰色 (0.5, 0.5, 0.5)
            // 完全忽略 p[12], p[13], p[14] 里的 SDK 垃圾值
            out->colors_.push_back({0.5, 0.5, 0.5});
        }
        return out;
    }

    vkc::PointCloudParams pcParams;
    std::string pointCloudDirPath;
    bool store_clouds;
    PointCloudGeneratorParams genParams;
    std::shared_ptr<PointCloudAccumulator> accumulator;
    double manualOffset;

    std::shared_ptr<ImageQueue> leftImgQ;
    std::shared_ptr<DisparityQueue> leftDispQ;
    
    std::map<double, Eigen::Isometry3d> optimized_poses_;
    std::shared_ptr<open3d::geometry::PointCloud> submap_cloud_;
};

// ===================== Part 4: Receivers =====================
template <typename T, typename Q>
class SimpleReceiver : public vkc::Receiver<T> {
public:
    SimpleReceiver(std::shared_ptr<Q> q) : queue(q) {}
    vkc::ReceiverStatus handle(const vkc::Message<vkc::Shared<T>> &msg) override {
        if(queue) queue->push(msg.payload);
        return vkc::ReceiverStatus::Open;
    }
private:
    std::shared_ptr<Q> queue;
};

// ===================== Part 5: Main =====================
int main(int argc, char *argv[]) {
    ProgramArgs args;
    PointCloudGeneratorParams genParams;
    CLI::App app;
    app.add_option("input", args.inputFile, "MCAP file")->capture_default_str();
    app.add_option("-o,--output", args.outputDir, "Output dir")->required();
    app.add_option("-c,--config", args.config_file, "Config")->required();
    app.add_option("--pose-file", args.poseFile, "Poses")->required();
    app.add_option("--manual-offset", args.manualOffset, "Offset");
    app.add_flag("--store-clouds", args.store_clouds);
    app.add_flag("--publish_clouds", args.publish_clouds);
    app.add_option("--output_topic", args.output_topic);
    app.add_option("-v,--device-version", args.deviceVersion);
    
    app.add_flag("--store-disparity", args.store_disparity);
    app.add_flag("--store-images", args.store_images);
    app.add_flag("--store-cumulative", args.store_cumulative);
    app.add_option("--input_cloud_topic", args.input_cloud_topic);
    app.add_option("--z-color-period", args.z_color_period);
    app.add_option("--disparity_suffix", args.disparity_suffix);
    app.add_option("-r,--playback-rate", args.playbackRate);

    CLI11_PARSE(app, argc, argv);

    if (args.config_file != "") {
        std::ifstream ifs(args.config_file);
        if(ifs.is_open()) {
            cereal::JSONInputArchive archive(ifs);
            archive(cereal::make_nvp("point_cloud_generator", genParams));
        }
    }

    std::shared_ptr<vkc::DataSource> source;
    auto visualkit = vkc::VisualKit::create(std::nullopt);
    if (args.inputFile != "") {
        mcap::McapReader bagReader;
        mcap::ReadMessageOptions opts;
        source = std::shared_ptr<vkc::DataSource>(vkc::McapSource::create(args.inputFile, opts, args.playbackRate));
    }
    vkc::DataSource* activeSource = source ? source.get() : &visualkit->source();

    std::shared_ptr<PointCloudAccumulator> acc = nullptr;
    if (args.publish_clouds) {
        auto rx = visualkit->sink().obtain(args.output_topic, vkc::Type<vkc::PointCloud>());
        acc = std::make_shared<PointCloudAccumulator>(args.outputDir, true, args.output_topic, std::move(rx));
    }

    vkc::PointCloudParams pcParams; 
    pcParams.maxDepth = genParams.maxDepth; pcParams.disparityOffset = genParams.disparityOffset;
    
    auto generator = std::make_shared<PointCloudGenerator>(
        pcParams, args.outputDir, args.store_clouds, genParams, acc, args.manualOffset
    );
    if (!args.poseFile.empty()) generator->loadOptimizedPoses(args.poseFile);

    std::cout << "Installing receivers..." << std::endl;
    activeSource->install("S1/stereo1_l", std::make_unique<SimpleReceiver<vkc::Image, ImageQueue>>(generator->getLeftImgQ()));
    activeSource->install("S1/stereo1_l/disparity", std::make_unique<SimpleReceiver<vkc::Disparity, DisparityQueue>>(generator->getLeftDispQ()));

    std::thread t([generator](){ generator->processStereoStream(); });
    t.detach();

    std::cout << "Running..." << std::endl;
    visualkit->sink().start();
    if (source) source->start();
    vkc::waitForCtrlCSignal();
    return 0;
}