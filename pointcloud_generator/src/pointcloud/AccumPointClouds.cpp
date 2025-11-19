#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <open3d/Open3D.h>
#include <CLI/CLI.hpp>

struct ProgramArgs {
    std::string inputFolder;
    std::string outputFile;
    std::string filePrefix = "";
    bool voxelDownsample;
    float voxelSize = 0.03;
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
    app.add_option("input", args.inputFolder, "Input directory containing .pcd files to combine")->required();
    app.add_option("output", args.outputFile, "Output file to save the combined point cloud")->required();
    app.add_option("--file_prefix", args.filePrefix, "Prefix of the .pcd files to combine (default: nothing)")->capture_default_str();
    app.add_option("--start", args.startFrame, "Start frame number");
    app.add_option("--end", args.endFrame, "End frame number");
    app.add_option("-i,--interval", args.skipInterval, "Number of frames to skip");
    app.add_option("--accumulate_interval", args.accumulateInterval, "Number of frames to accumulate before processing");
    app.add_flag("--voxel-downsample", args.voxelDownsample, "Downsample the point cloud using voxel grid filtering");
    app.add_option("--voxel-size", args.voxelSize, "Voxel size for voxel grid filtering");
    app.add_flag("--statistical-outlier-removal", args.statisticalOutlierRemoval, "Apply statistical outlier removal to the point cloud");
    app.add_option("--sor-neighbors", args.sorNeighbors, "Number of neighbors to consider for statistical outlier removal");
    app.add_option("--sor-stddev", args.sorStddev, "Std dev to consider for statistical outlier removal");
    app.add_flag("--radius-outlier-removal", args.radiusOutlierRemoval, "Apply radius outlier removal to the point cloud");
    app.add_option("--ror-points", args.rorPoints, "Number of points to consider for radius outlier removal");
    app.add_option("--ror-radius", args.rorRadius, "Radius to consider for radius outlier removal");
    CLI11_PARSE(app, argc, argv);

    float ror_duration = 0.0;
    float sor_duration = 0.0;
    float voxel_downsample_duration = 0.0;
    int num_frames = 0;

    // Load all .pcd files in the input directory
    std::vector<std::string> pcd_files;
    std::map<int, std::vector<std::string>> pcd_files_map;
    std::cout << "Loading .pcd files from " << args.inputFolder << std::endl;
    for (const auto& entry : std::filesystem::directory_iterator(args.inputFolder)) {
        if (entry.path().extension() == ".pcd") {
            auto filename = entry.path().stem().string();
            if (!args.filePrefix.empty() && filename.find(args.filePrefix) == std::string::npos) {
                continue;
            }
            auto index = filename.find_last_of("_");
            auto frame = std::stoi(filename.substr(index + 1));
            pcd_files_map[frame].push_back(entry.path().string());
        }
    }

    // Combine point clouds into a single point cloud
    auto combinedCloud = std::make_shared<open3d::geometry::PointCloud>();
    auto cloud = std::make_shared<open3d::geometry::PointCloud>();
    for (const auto& [frame, files] : pcd_files_map) {
        if (frame < args.startFrame || (args.endFrame != -1 && frame > args.endFrame)) {
            continue;
        }
        if (frame % args.skipInterval != 0) {
            continue;
        }
        num_frames++;
        std::cout << "Accumulating frame " << frame << std::endl;
        for (const auto& file : files) {
            auto frameCloud = open3d::io::CreatePointCloudFromFile(file);
            // for (const auto& pixel : cloud->colors_)
            // {
            //     std::cout << "rgb:" << pixel.x() << "," << pixel.y() << "," << pixel.z() << std::endl;
            // }
            if (!frameCloud) {
                std::cerr << "Couldn't read file " << file << std::endl;
                continue;
            }
            *cloud += *frameCloud;
        }
        if (cloud->IsEmpty()) {
            std::cerr << "No point clouds to accumulate for frame " << frame << std::endl;
            continue;
        }

        if (frame % (args.accumulateInterval * args.skipInterval) != 0) {
            continue;
        }

        // process accumulated cloud
        if (args.voxelDownsample) {
            auto start = std::chrono::high_resolution_clock::now();
            cloud = cloud->VoxelDownSample(args.voxelSize);
            std::cout << "Voxel downsampled point cloud with " << cloud->points_.size() << " points\n";
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            voxel_downsample_duration += duration.count();
        }

        if (args.radiusOutlierRemoval) {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = cloud->RemoveRadiusOutliers(args.rorPoints, args.rorRadius);

            std::cout << "Radius outlier removal applied to point cloud with " << cloud->points_.size() << " points\n";
            cloud = std::get<0>(result);
            std::cout << "Cloud size after ROR: " << cloud->points_.size() << std::endl;
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            ror_duration += duration.count();
        }

        if (args.statisticalOutlierRemoval) {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = cloud->RemoveStatisticalOutliers(args.sorNeighbors, 2.0);

            std::cout << "Statistical outlier removal applied to point cloud with " << cloud->points_.size() << " points\n";
            cloud = std::get<0>(result);
            std::cout << "Cloud size after SOR: " << cloud->points_.size() << std::endl;
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            sor_duration += duration.count();
        }

        *combinedCloud += *cloud;
        cloud->Clear();
    }

    if (args.voxelDownsample) {
        auto initialSize = combinedCloud->points_.size();
        combinedCloud = combinedCloud->VoxelDownSample(args.voxelSize);
        std::cout << "Point cloud downsampled from " << initialSize << " to " << combinedCloud->points_.size() << " points\n";
    }

    std::cout << "Total number of frames processed: " << num_frames << std::endl;
    std::cout << "Total number of points in combined point cloud: " << combinedCloud->points_.size() << std::endl;
    std::cout << "Average downsample duration: " << voxel_downsample_duration / num_frames << "us" << std::endl;
    std::cout << "Average radius outlier removal duration: " << ror_duration / num_frames << "us" << std::endl;
    std::cout << "Average statistical outlier removal duration: " << sor_duration / num_frames << "us" << std::endl;

    // Save the combined point cloud
    // const std::vector<std::shared_ptr<const open3d::geometry::Geometry>> geometries = {
    //     combinedCloud
    // };
    
    // open3d::visualization::DrawGeometries(geometries, "Point Cloud");

    // Save the combined point cloud
    if (!open3d::io::WritePointCloud(args.outputFile, *combinedCloud)) {
        std::cerr << "Error: Failed to write combined point cloud!" << std::endl;
        return 1;
    }
    std::cout << "Combined point cloud saved to " << args.outputFile << std::endl;

    return 0;
}
