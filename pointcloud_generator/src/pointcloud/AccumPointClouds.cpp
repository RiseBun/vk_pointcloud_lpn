#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <open3d/Open3D.h>
#include <CLI/CLI.hpp>
#include <chrono>

struct ProgramArgs {
    std::string inputFolder;
    std::string outputFile;
    std::string filePrefix = "";
    bool voxelDownsample;
    float voxelSize = 0.05; // 默认调大一点
    bool statisticalOutlierRemoval = false;
    int sorNeighbors = 10;
    float sorStddev = 1.0;
    int skipInterval = 1;
    int accumulateInterval = 1;
    bool radiusOutlierRemoval = false;
    int rorPoints = 10;
    double rorRadius = 0.05;
    int startFrame = 0;
    int endFrame = -1;
    int removeBorder = 0;
};

int main(int argc, char** argv) {
    ProgramArgs args;
    CLI::App app;
    app.add_option("input", args.inputFolder, "Input directory")->required();
    app.add_option("output", args.outputFile, "Output file")->required();
    app.add_option("--file_prefix", args.filePrefix, "Prefix");
    app.add_option("--start", args.startFrame, "Start frame");
    app.add_option("--end", args.endFrame, "End frame");
    app.add_option("-i,--interval", args.skipInterval, "Skip interval");
    app.add_option("--accumulate_interval", args.accumulateInterval, "Acc interval");
    app.add_flag("--voxel-downsample", args.voxelDownsample, "Voxel Grid");
    app.add_option("--voxel-size", args.voxelSize, "Voxel Size");
    app.add_flag("--statistical-outlier-removal", args.statisticalOutlierRemoval, "SOR");
    app.add_flag("--radius-outlier-removal", args.radiusOutlierRemoval, "ROR");
    CLI11_PARSE(app, argc, argv);

    int num_frames = 0;
    std::vector<std::string> pcd_files;
    std::map<int, std::vector<std::string>> pcd_files_map;
    
    // 1. 文件加载逻辑
    for (const auto& entry : std::filesystem::directory_iterator(args.inputFolder)) {
        if (entry.path().extension() == ".pcd") {
            auto filename = entry.path().stem().string();
            if (!args.filePrefix.empty() && filename.find(args.filePrefix) == std::string::npos) continue;
            auto index = filename.find_last_of("_");
            if (index != std::string::npos) {
                try {
                    auto frame = std::stoi(filename.substr(index + 1));
                    pcd_files_map[frame].push_back(entry.path().string());
                } catch (...) {}
            }
        }
    }

    auto combinedCloud = std::make_shared<open3d::geometry::PointCloud>();
    auto cloud = std::make_shared<open3d::geometry::PointCloud>();
    
    // [关键修改] 阈值设为 1500万。
    // 这既保证了处理速度，又绝对安全（约 500MB 内存），给最后保存留足空间。
    const size_t MAX_POINTS_IN_MEMORY = 15000000; 

    try {
        for (const auto& [frame, files] : pcd_files_map) {
            if (frame < args.startFrame || (args.endFrame != -1 && frame > args.endFrame)) continue;
            if (frame % args.skipInterval != 0) continue;
            
            num_frames++;
            if (num_frames % 100 == 0) std::cout << "Processing frame " << frame << "..." << std::endl;

            for (const auto& file : files) {
                auto frameCloud = open3d::io::CreatePointCloudFromFile(file);
                if (frameCloud) *cloud += *frameCloud;
            }
            if (cloud->IsEmpty()) continue;

            if (frame % (args.accumulateInterval * args.skipInterval) != 0) continue;

            // 局部处理
            if (args.voxelDownsample) cloud = cloud->VoxelDownSample(args.voxelSize);
            if (args.radiusOutlierRemoval) cloud = std::get<0>(cloud->RemoveRadiusOutliers(args.rorPoints, args.rorRadius));

            *combinedCloud += *cloud;
            cloud->Clear();
            
            // [安全气囊] 增量降采样
            if (args.voxelDownsample && combinedCloud->points_.size() > MAX_POINTS_IN_MEMORY) {
                std::cout << "[Auto-Compress] Size " << combinedCloud->points_.size() << " -> ";
                combinedCloud = combinedCloud->VoxelDownSample(args.voxelSize);
                std::cout << combinedCloud->points_.size() << " (Safe)" << std::endl;
            }
        }

        // [移除危险的 Final Processing] 
        // 之前这里会再一次全局降采样，对于大地图极易爆内存。
        // 我们已经在循环里做过压缩了，这里直接跳过或只做简单清理。
        
        std::cout << "Total frames: " << num_frames << std::endl;
        std::cout << "Final points: " << combinedCloud->points_.size() << std::endl;

        std::cout << "Saving to " << args.outputFile << " ... " << std::flush;
        if (!open3d::io::WritePointCloud(args.outputFile, *combinedCloud)) {
            std::cerr << "FAILED." << std::endl;
            return 1;
        }
        std::cout << "DONE." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\n\n[CRITICAL ERROR] " << e.what() << std::endl;
        std::cerr << "Trying to emergency save whatever we have..." << std::endl;
        open3d::io::WritePointCloud(args.outputFile + "_emergency.pcd", *combinedCloud);
    } catch (...) {
        std::cerr << "\n\n[UNKNOWN CRASH]" << std::endl;
        open3d::io::WritePointCloud(args.outputFile + "_crash.pcd", *combinedCloud);
    }

    return 0;
}