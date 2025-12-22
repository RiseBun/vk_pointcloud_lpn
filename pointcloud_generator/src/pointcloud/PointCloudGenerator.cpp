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
#include <numeric> // For statistics

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

// ===================== Part 1: 结构体 =====================
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
    float maxDepth = 4.5; // Sharp Mode
    bool preVoxelDownsample = true;
    float preVoxelSize = 0.03; 
    bool voxelDownsample = true;
    float voxelSize = 0.03;
    int sorNeighbors = 20;
    float sorStddev = 0.8; 
    int accumulateInterval = 5; 
    bool radiusOutlierRemoval = true; 
    int rorPoints = 6;                
    double rorRadius = 0.10;          
    bool remove_statistical_outlier = true;
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

// ===================== Part 2: 辅助 =====================
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
            bool has_valid_colors = cloud->HasColors() && (cloud->colors_.size() == cloud->points_.size());
            auto mmb = std::make_unique<capnp::MallocMessageBuilder>();
            vkc::PointCloud::Builder msg = mmb->getRoot<vkc::PointCloud>();
            int n = cloud->points_.size();
            msg.setPointStride(16);
            msg.initPoints(n * 16);
            unsigned char *ptr = msg.getPoints().asBytes().begin();
            for (int i=0; i<n; ++i) {
                float *p = reinterpret_cast<float*>(ptr);
                p[0] = cloud->points_[i].x(); p[1] = cloud->points_[i].y(); p[2] = cloud->points_[i].z();
                ptr[12] = 0; ptr[13] = 0; ptr[14] = 0; ptr[15] = 0; 
                if (has_valid_colors) {
                    ptr[12] = (uint8_t)(cloud->colors_[i].x()*255); 
                    ptr[13] = (uint8_t)(cloud->colors_[i].y()*255); 
                    ptr[14] = (uint8_t)(cloud->colors_[i].z()*255); 
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

// ===================== Part 3: PointCloudGenerator (Diagnostic Version) =====================
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
        std::cout << "[Config] Max Depth: " << pcParams.maxDepth << "m" << std::endl;
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
        
        double auto_offset = 0.0;
        bool time_aligned = false;

        // Diagnostic Variables
        Eigen::Isometry3d prev_diag_pose = Eigen::Isometry3d::Identity();
        bool first_diag = true;

        std::cout << "[INFO] Processing started (Diagnostic Mode)..." << std::endl;

        while (true) {
            vkc::Shared<vkc::Image> i; while(leftImgQ->try_pop(i)) lImgBuf.push_back(i);
            vkc::Shared<vkc::Disparity> d; while(leftDispQ->try_pop(d)) lDispBuf.push_back(d);

            if (lImgBuf.empty()) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); continue; }
            
            auto currLImg = lImgBuf.front();
            auto imgHeader = currLImg.reader().getHeader();
            uint64_t target_ns = imgHeader.getStampMonotonic(); 
            double target_sec = target_ns * 1e-9;
            
            if (!time_aligned && !optimized_poses_.empty()) {
                double first_pose_t = optimized_poses_.begin()->first;
                if (std::abs(target_sec - first_pose_t) > 5.0) {
                    auto_offset = first_pose_t - target_sec;
                    std::cout << "[AUTO-ALIGN] Offset: " << auto_offset << "s" << std::endl;
                }
                time_aligned = true;
            }

            auto currLDispOpt = findAndConsumeNearest(lDispBuf, target_ns, 10000000); 
            if (!currLDispOpt) {
                if (lImgBuf.size() > 50) lImgBuf.pop_front(); 
                continue; 
            }
            auto currLDisp = *currLDispOpt;
            lImgBuf.pop_front(); 

            double query_time = target_sec + manualOffset + auto_offset;
            Eigen::Isometry3d T_world_body; 
            
            if (!getOptimizedPose(query_time, T_world_body)) continue; 
            
            // [DIAGNOSTICS] 统计抖动 (Jitter)
            if (first_diag) {
                prev_diag_pose = T_world_body;
                first_diag = false;
            } else {
                double dt = (T_world_body.translation() - prev_diag_pose.translation()).norm();
                double da = Eigen::AngleAxisd(prev_diag_pose.linear().inverse() * T_world_body.linear()).angle();
                stat_trans_deltas.push_back(dt * 1000.0); // mm
                stat_rot_deltas.push_back(da * 57.2958); // degree
                prev_diag_pose = T_world_body;
            }

            // [DIAGNOSTICS] 定期打印报告
            if (frameCount > 0 && frameCount % 100 == 0) {
                printDiagnostics(frameCount);
            }

            if (first_frame) {
                last_keyframe_pose = T_world_body;
                first_frame = false;
                std::cout << "[SUCCESS] First frame matched at t=" << std::fixed << query_time << std::endl;
            } 
            
            // 关掉关键帧跳过
            last_keyframe_pose = T_world_body;

            // 心跳点
            // std::cout << "." << std::flush; 

            auto frameCloud = generateCloudCam(currLDisp, currLImg);
            
            if (frameCloud && !frameCloud->IsEmpty()) {
                Eigen::Isometry3d T_body_cam = eCALSe3toEigen(currLImg.reader().getExtrinsic().getBodyFrame());
                Eigen::Isometry3d T_total = T_world_body * T_body_cam;
                frameCloud->Transform(T_total.matrix()); 
                *submap_cloud_ += *frameCloud;
                submapCount++;
                submapTimes.push_back(query_time);
            }
            frameCount++;

            if (submapCount >= genParams.accumulateInterval) {
                if (!submap_cloud_->IsEmpty()) {
                    submap_cloud_ = submap_cloud_->VoxelDownSample(genParams.voxelSize);
                    if (genParams.remove_statistical_outlier) {
                        auto res = submap_cloud_->RemoveStatisticalOutliers(genParams.sorNeighbors, genParams.sorStddev);
                        submap_cloud_ = std::get<0>(res);
                    }
                    if (genParams.radiusOutlierRemoval) {
                        auto res = submap_cloud_->RemoveRadiusOutliers(genParams.rorPoints, genParams.rorRadius);
                        submap_cloud_ = std::get<0>(res);
                    }

                    double mid_t = submapTimes[submapTimes.size()/2];
                    std::string path = pointCloudDirPath + "/" + std::to_string((uint64_t)(mid_t*1e9)) + "_submap_" + std::to_string(frameCount) + ".pcd";
                    
                    if (store_clouds) open3d::io::WritePointCloud(path, *submap_cloud_);
                    if (accumulator) accumulator->process(submap_cloud_, currLImg.reader().getHeader());
                }
                submap_cloud_->Clear();
                submapCount = 0;
                submapTimes.clear();
            }
        }
    }

private:
    // 统计容器
    std::vector<double> stat_time_errors; // ms
    std::vector<double> stat_trans_deltas; // mm
    std::vector<double> stat_rot_deltas;   // deg

    void printDiagnostics(int frameCount) {
        if (stat_time_errors.empty()) return;

        // 计算 Time Error Stats
        double sum_t = std::accumulate(stat_time_errors.begin(), stat_time_errors.end(), 0.0);
        double mean_t = sum_t / stat_time_errors.size();
        std::vector<double> sorted_t = stat_time_errors;
        std::sort(sorted_t.begin(), sorted_t.end());
        double p99_t = sorted_t[(int)(sorted_t.size() * 0.99)];

        // 计算 Jitter Stats (Translation)
        double mean_trans = 0, p99_trans = 0;
        if (!stat_trans_deltas.empty()) {
            double sum = std::accumulate(stat_trans_deltas.begin(), stat_trans_deltas.end(), 0.0);
            mean_trans = sum / stat_trans_deltas.size();
            std::vector<double> s = stat_trans_deltas;
            std::sort(s.begin(), s.end());
            p99_trans = s[(int)(s.size() * 0.99)];
        }

        // 计算 Jitter Stats (Rotation)
        double mean_rot = 0, p99_rot = 0;
        if (!stat_rot_deltas.empty()) {
            double sum = std::accumulate(stat_rot_deltas.begin(), stat_rot_deltas.end(), 0.0);
            mean_rot = sum / stat_rot_deltas.size();
            std::vector<double> s = stat_rot_deltas;
            std::sort(s.begin(), s.end());
            p99_rot = s[(int)(s.size() * 0.99)];
        }

        std::cout << "\n=== [DIAGNOSTICS FRAME " << frameCount << "] ===" << std::endl;
        std::cout << " [TimeSync] Lag (img vs pose): Mean=" << std::fixed << std::setprecision(2) << mean_t 
                  << "ms | 99%=" << p99_t << "ms " 
                  << (p99_t > 15.0 ? "\033[1;31m[BAD]\033[0m" : "\033[1;32m[OK]\033[0m") << std::endl;
        
        std::cout << " [PoseJitter] Trans Delta: Mean=" << mean_trans 
                  << "mm | 99%=" << p99_trans << "mm" << std::endl;
        
        std::cout << " [PoseJitter] Rot Delta:   Mean=" << mean_rot 
                  << "deg | 99%=" << p99_rot << "deg" << std::endl;
        std::cout << "======================================\n" << std::endl;

        // 清空以便统计下一批
        stat_time_errors.clear();
        stat_trans_deltas.clear();
        stat_rot_deltas.clear();
    }

    template <typename T>
    std::optional<vkc::Shared<T>> findAndConsumeNearest(std::deque<vkc::Shared<T>>& buf, uint64_t target_ns, int64_t tol_ns) {
        std::optional<vkc::Shared<T>> best = std::nullopt;
        int64_t min_diff = tol_ns + 1;
        auto it = buf.begin();
        auto it_best = buf.end();

        while (it != buf.end()) {
            uint64_t item_ts = it->reader().getHeader().getStampMonotonic();
            int64_t diff = (int64_t)item_ts - (int64_t)target_ns;
            if (diff < -tol_ns) { it++; continue; }
            if (diff > tol_ns) break;
            
            int64_t abs_diff = std::abs(diff);
            if (abs_diff < min_diff) {
                min_diff = abs_diff;
                best = *it;
                it_best = it;
            }
            it++;
        }

        if (best && it_best != buf.end()) {
            buf.erase(buf.begin(), std::next(it_best)); 
        } else {
            while(!buf.empty()) {
                uint64_t ts = buf.front().reader().getHeader().getStampMonotonic();
                if ((int64_t)target_ns - (int64_t)ts > 100000000) buf.pop_front();
                else break;
            }
        }
        return best;
    }

    bool getOptimizedPose(double t, Eigen::Isometry3d& out) {
        if (optimized_poses_.empty()) return false;
        auto it = optimized_poses_.lower_bound(t);
        
        if (it == optimized_poses_.begin()) { 
            if (std::abs(it->first - t) > 0.05) return false;
            out = it->second; return true; 
        }
        if (it == optimized_poses_.end()) { 
            auto prev = std::prev(it);
            if (std::abs(prev->first - t) > 0.05) return false;
            out = prev->second; return true; 
        }
        
        auto prev = std::prev(it);
        
        // [DIAGNOSTICS] 记录查找误差
        double dist_to_nearest = std::min(std::abs(t - prev->first), std::abs(t - it->first));
        stat_time_errors.push_back(dist_to_nearest * 1000.0); // ms

        if (std::abs(it->first - prev->first) > 2.0) return false; 

        double alpha = (t - prev->first)/(it->first - prev->first);
        out.translation() = prev->second.translation() * (1-alpha) + it->second.translation() * alpha;
        Eigen::Quaterniond qprev(prev->second.rotation());
        Eigen::Quaterniond qnext(it->second.rotation());
        out.linear() = qprev.slerp(alpha, qnext).normalized().toRotationMatrix();
        return true;
    }

    std::shared_ptr<open3d::geometry::PointCloud> generateCloudCam(vkc::Shared<vkc::Disparity> disp, vkc::Shared<vkc::Image> img) {
        auto iR = img.reader();
        cv::Mat bgr_mat; 
        bool decode_success = false;
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
                if (temp.channels() == 1) cv::cvtColor(temp, bgr_mat, cv::COLOR_GRAY2BGR);
                else bgr_mat = temp;
                if (!bgr_mat.isContinuous()) bgr_mat = bgr_mat.clone();
                decode_success = true;
            }
        } catch (...) { decode_success = false; }

        if (!decode_success) return std::make_shared<open3d::geometry::PointCloud>();

        auto vkCloud = vkc::convertToPointCloud(disp, bgr_mat.data, pcParams);
        auto out = std::make_shared<open3d::geometry::PointCloud>();
        auto ptsReader = vkCloud.reader().getPoints();
        auto ptsData = ptsReader.asBytes();
        
        int stride = (int)vkCloud.reader().getPointStride();
        if (stride <= 0) stride = 16;
        size_t count = ptsData.size() / stride;
        const uint8_t* base_ptr = ptsData.begin();

        if (ptsData.size() == 0 || base_ptr == nullptr) return out;

        for(size_t i = 0; i < count; ++i) {
            const uint8_t* p = base_ptr + i * stride;
            float x, y, z;
            std::memcpy(&x, p + 0, 4);
            std::memcpy(&y, p + 4, 4);
            std::memcpy(&z, p + 8, 4);
            
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
            float r_sq = x*x + y*y + z*z;
            
            if (r_sq < 0.25f || r_sq > pcParams.maxDepth * pcParams.maxDepth) continue;
            if (std::abs(x) > 20.0f || std::abs(y) > 20.0f) continue;

            out->points_.push_back({(double)x, (double)y, (double)z});
            
            uint8_t r = p[12], g = p[13], b = p[14];
            out->colors_.push_back({r/255.0, g/255.0, b/255.0});
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
    SimpleReceiver(std::shared_ptr<Q> q, std::string topic_name) : queue(q), name(topic_name) {}
    vkc::ReceiverStatus handle(const vkc::Message<vkc::Shared<T>> &msg) override {
        if(queue) queue->push(msg.payload);
        return vkc::ReceiverStatus::Open;
    }
private:
    std::shared_ptr<Q> queue;
    std::string name;
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
    
    std::string topic_img = "S1/stereo1_l";
    std::string topic_disp = "S1/stereo1_l/disparity";
    std::cout << "[Config] Listening to Image: " << topic_img << std::endl;
    std::cout << "[Config] Listening to Disparity: " << topic_disp << std::endl;

    activeSource->install(topic_img, std::make_unique<SimpleReceiver<vkc::Image, ImageQueue>>(generator->getLeftImgQ(), "Image"));
    activeSource->install(topic_disp, std::make_unique<SimpleReceiver<vkc::Disparity, DisparityQueue>>(generator->getLeftDispQ(), "Disparity"));

    std::thread t([generator](){ generator->processStereoStream(); });
    t.detach();

    std::cout << "Running..." << std::endl;
    visualkit->sink().start();
    if (source) source->start();
    vkc::waitForCtrlCSignal();
    return 0;
}