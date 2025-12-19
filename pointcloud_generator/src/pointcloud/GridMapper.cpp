#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_unordered_map.h>
#include <open3d/Open3D.h>
#include <ecal/ecal.h>
#include <spdlog/spdlog.h>

#include <fstream>
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

#include "pointcloud/GridMapper.hpp"


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

void GridMap::initMap(GridMapperParams &cfg)
{

    visualkit = vkc::VisualKit::create(std::nullopt);

    if (visualkit == nullptr)
    {
        std::cout << "Failed to create VisualKit connection." << std::endl;
        return;
    }
    vkc_source_ref_ = std::ref(visualkit->source());
    output_topic_ = cfg.tf_prefix + cfg.output_topic_;
    map_receiver = visualkit->sink().obtain(output_topic_, vkc::Type<vkc::PointCloud>());
    map_inflate_receiver = visualkit->sink().obtain(output_topic_ + "_inflate", vkc::Type<vkc::PointCloud>());

    /* get parameter */
    double x_size, y_size, z_size;
    x_size = cfg.map_size_x_;
    y_size = cfg.map_size_y_;
    z_size = cfg.map_size_z_;
    mp_.local_update_range_(0) = cfg.local_update_range_x_;
    mp_.local_update_range_(1) = cfg.local_update_range_y_;
    mp_.local_update_range_(2) = cfg.local_update_range_z_;
    mp_.obstacles_inflation_ = cfg.obstacles_inflation_;
    mp_.p_hit_ = cfg.p_hit_;
    mp_.p_miss_ = cfg.p_miss_;
    mp_.p_min_ = cfg.p_min_;
    mp_.p_max_ = cfg.p_max_;
    mp_.p_occ_ = cfg.p_occ_;
    mp_.min_ray_length_ = cfg.min_ray_length_;
    mp_.max_ray_length_ = cfg.max_ray_length_;
    mp_.local_map_margin_ = cfg.local_map_margin_;
    mp_.odom_depth_timeout_ = cfg.odom_depth_timeout_;
    mp_.resolution_ = cfg.resolution_;
    mp_.skip_pixel_ = cfg.skip_pixel_;
    mp_.visualization_truncate_height_ = cfg.visualization_truncate_height_;
    mp_.fading_time_ = cfg.fading_time_;

    mp_.resolution_inv_ = 1 / mp_.resolution_;
    mp_.map_origin_ = Eigen::Vector3d(-x_size / 2.0, -y_size / 2.0, -z_size / 2.0);
    mp_.map_size_ = Eigen::Vector3d(x_size, y_size, z_size);

    mp_.prob_hit_log_ = logit(mp_.p_hit_);
    mp_.prob_miss_log_ = logit(mp_.p_miss_);
    mp_.clamp_min_log_ = logit(mp_.p_min_);
    mp_.clamp_max_log_ = logit(mp_.p_max_);
    mp_.min_occupancy_log_ = logit(mp_.p_occ_);
    mp_.unknown_flag_ = 0.01;

    cout << "hit: " << mp_.prob_hit_log_ << endl;
    cout << "miss: " << mp_.prob_miss_log_ << endl;
    cout << "min log: " << mp_.clamp_min_log_ << endl;
    cout << "max: " << mp_.clamp_max_log_ << endl;
    cout << "thresh log: " << mp_.min_occupancy_log_ << endl;

    for (int i = 0; i < 3; ++i)
        mp_.map_voxel_num_(i) = ceil(mp_.map_size_(i) / mp_.resolution_);

    mp_.map_min_boundary_ = mp_.map_origin_;
    mp_.map_max_boundary_ = mp_.map_origin_ + mp_.map_size_;
    mp_.map_remain_min_boundary_ = mp_.map_origin_ + mp_.map_size_ / 8.0;
    mp_.map_remain_max_boundary_ = mp_.map_origin_ + 7 * mp_.map_size_ / 8.0;

    // initialize data buffers

    int buffer_size = mp_.map_voxel_num_(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2);

    md_.occupancy_buffer_ = vector<double>(buffer_size, mp_.clamp_min_log_ - mp_.unknown_flag_);
    md_.occupancy_buffer_inflate_ = vector<char>(buffer_size, 0);

    md_.count_hit_and_miss_ = vector<short>(buffer_size, 0);
    md_.count_hit_ = vector<short>(buffer_size, 0);
    md_.flag_rayend_ = vector<char>(buffer_size, -1);
    md_.flag_traverse_ = vector<char>(buffer_size, -1);

    md_.raycast_num_ = 0;

    md_.proj_points_.resize(640 * 480);
    md_.proj_points_cnt = 0;

    md_.cam2body_ << 0.0, 0.0, 1.0, 0.0,
        -1.0, 0.0, 0.0, 0.0,
        0.0, -1.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0;

    /* init callback */
    md_.occ_need_update_ = false;
    md_.local_updated_ = false;
    md_.has_first_depth_ = false;
    md_.has_odom_ = false;
    md_.has_cloud_ = false;
    md_.image_cnt_ = 0;
    md_.last_occ_update_time_ = 0;

    md_.fuse_time_ = 0.0;
    md_.update_num_ = 0;
    md_.max_fuse_time_ = 0.0;

    md_.flag_depth_odom_timeout_ = false;
    md_.flag_use_depth_fusion = false;

    // rand_noise_ = uniform_real_distribution<double>(-0.2, 0.2);
    // rand_noise2_ = normal_distribution<double>(0, 0.2);
    // random_device rd;
    // eng_ = default_random_engine(rd());
}

void GridMap::resetBuffer()
{
    Eigen::Vector3d min_pos = mp_.map_min_boundary_;
    Eigen::Vector3d max_pos = mp_.map_max_boundary_;

    resetBuffer(min_pos, max_pos);

    md_.local_bound_min_ = Eigen::Vector3i::Zero();
    md_.local_bound_max_ = mp_.map_voxel_num_ - Eigen::Vector3i::Ones();
}

void GridMap::resetBuffer(Eigen::Vector3d min_pos, Eigen::Vector3d max_pos)
{

    Eigen::Vector3i min_id, max_id;
    posToIndex(min_pos, min_id);
    posToIndex(max_pos, max_id);

    boundIndex(min_id);
    boundIndex(max_id);

    /* reset occ and dist buffer */
    for (int x = min_id(0); x <= max_id(0); ++x)
        for (int y = min_id(1); y <= max_id(1); ++y)
            for (int z = min_id(2); z <= max_id(2); ++z)
            {
                md_.occupancy_buffer_inflate_[toAddress(x, y, z)] = 0;
            }
}

int GridMap::setCacheOccupancy(Eigen::Vector3d pos, int occ)
{
    if (occ != 1 && occ != 0)
        return INVALID_IDX;

    Eigen::Vector3i id;
    posToIndex(pos, id);
    int idx_ctns = toAddress(id);

    md_.count_hit_and_miss_[idx_ctns] += 1;

    if (md_.count_hit_and_miss_[idx_ctns] == 1)
    {
        md_.cache_voxel_.push(id);
    }

    if (occ == 1)
        md_.count_hit_[idx_ctns] += 1;

    return idx_ctns;
}

void GridMap::raycastProcess()
{
    // if (md_.proj_points_.size() == 0)
    if (md_.proj_points_cnt == 0) {
        spdlog::info("No projected points, skip raycasting");
        return;
    }

    //   ros::Time t1, t2;

    md_.raycast_num_ += 1;

    int vox_idx;
    double length;

    // bounding box of updated region
    double min_x = mp_.map_max_boundary_(0);
    double min_y = mp_.map_max_boundary_(1);
    double min_z = mp_.map_max_boundary_(2);

    double max_x = mp_.map_min_boundary_(0);
    double max_y = mp_.map_min_boundary_(1);
    double max_z = mp_.map_min_boundary_(2);

    RayCaster raycaster;
    Eigen::Vector3d half = Eigen::Vector3d(0.5, 0.5, 0.5);
    Eigen::Vector3d ray_pt, pt_w;
    for (int i = 0; i < md_.proj_points_cnt; ++i)
    {
        pt_w = md_.proj_points_[i];

        // set flag for projected point

        if (!isInMap(pt_w))
        {
            pt_w = closestPointInMap(pt_w, md_.camera_pos_);

            length = (pt_w - md_.camera_pos_).norm();
            if (length > mp_.max_ray_length_)
            {
                pt_w = (pt_w - md_.camera_pos_) / length * mp_.max_ray_length_ + md_.camera_pos_;
            }
            vox_idx = setCacheOccupancy(pt_w, 0);
        }
        else
        {
            length = (pt_w - md_.camera_pos_).norm();

            if (length > mp_.max_ray_length_)
            {
                pt_w = (pt_w - md_.camera_pos_) / length * mp_.max_ray_length_ + md_.camera_pos_;
                vox_idx = setCacheOccupancy(pt_w, 0);
            }
            else
            {
                vox_idx = setCacheOccupancy(pt_w, 1);
            }
        }

        max_x = max(max_x, pt_w(0));
        max_y = max(max_y, pt_w(1));
        max_z = max(max_z, pt_w(2));

        min_x = min(min_x, pt_w(0));
        min_y = min(min_y, pt_w(1));
        min_z = min(min_z, pt_w(2));

        // raycasting between camera center and point

        if (vox_idx != INVALID_IDX)
        {
            if (md_.flag_rayend_[vox_idx] == md_.raycast_num_)
            {
                continue;
            }
            else
            {
                md_.flag_rayend_[vox_idx] = md_.raycast_num_;
            }
        }

        raycaster.setInput(pt_w / mp_.resolution_, md_.camera_pos_ / mp_.resolution_);

        while (raycaster.step(ray_pt))
        {
            Eigen::Vector3d tmp = (ray_pt + half) * mp_.resolution_;
            length = (tmp - md_.camera_pos_).norm();

            if (length < mp_.min_ray_length_) break;

            vox_idx = setCacheOccupancy(tmp, 0);

            if (vox_idx != INVALID_IDX)
            {
                if (md_.flag_traverse_[vox_idx] == md_.raycast_num_)
                {
                    break;
                }
                else
                {
                    md_.flag_traverse_[vox_idx] = md_.raycast_num_;
                }
            }
        }
    }

    min_x = min(min_x, md_.camera_pos_(0));
    min_y = min(min_y, md_.camera_pos_(1));
    min_z = min(min_z, md_.camera_pos_(2));

    max_x = max(max_x, md_.camera_pos_(0));
    max_y = max(max_y, md_.camera_pos_(1));
    max_z = max(max_z, md_.camera_pos_(2));
    max_z = max(max_z, mp_.ground_height_);

    posToIndex(Eigen::Vector3d(max_x, max_y, max_z), md_.local_bound_max_);
    posToIndex(Eigen::Vector3d(min_x, min_y, min_z), md_.local_bound_min_);
    boundIndex(md_.local_bound_min_);
    boundIndex(md_.local_bound_max_);

    md_.local_updated_ = true;

    // update occupancy cached in queue
    Eigen::Vector3d local_range_min = md_.camera_pos_ - mp_.local_update_range_;
    Eigen::Vector3d local_range_max = md_.camera_pos_ + mp_.local_update_range_;

    Eigen::Vector3i min_id, max_id;
    posToIndex(local_range_min, min_id);
    posToIndex(local_range_max, max_id);
    boundIndex(min_id);
    boundIndex(max_id);

    update_map_mutex_.lock();
    while (!md_.cache_voxel_.empty())
    {

        Eigen::Vector3i idx = md_.cache_voxel_.front();
        int idx_ctns = toAddress(idx);
        md_.cache_voxel_.pop();

        double log_odds_update =
            md_.count_hit_[idx_ctns] >= md_.count_hit_and_miss_[idx_ctns] - md_.count_hit_[idx_ctns] ? mp_.prob_hit_log_ : mp_.prob_miss_log_;

        md_.count_hit_[idx_ctns] = md_.count_hit_and_miss_[idx_ctns] = 0;

        if (log_odds_update >= 0 && md_.occupancy_buffer_[idx_ctns] >= mp_.clamp_max_log_)
        {
            continue;
        }
        else if (log_odds_update <= 0 && md_.occupancy_buffer_[idx_ctns] <= mp_.clamp_min_log_)
        {
            md_.occupancy_buffer_[idx_ctns] = mp_.clamp_min_log_;
            continue;
        }

        // bool in_local = idx(0) >= min_id(0) && idx(0) <= max_id(0) && idx(1) >= min_id(1) &&
        //                 idx(1) <= max_id(1) && idx(2) >= min_id(2) && idx(2) <= max_id(2);
        // if (!in_local)
        // {
        //     md_.occupancy_buffer_[idx_ctns] = mp_.clamp_min_log_;
        // }

        md_.occupancy_buffer_[idx_ctns] =
            std::min(std::max(md_.occupancy_buffer_[idx_ctns] + log_odds_update, mp_.clamp_min_log_),
                     mp_.clamp_max_log_);
    }
    update_map_mutex_.unlock();
}

Eigen::Vector3d GridMap::closestPointInMap(const Eigen::Vector3d &pt, const Eigen::Vector3d &camera_pt)
{
    Eigen::Vector3d diff = pt - camera_pt;
    Eigen::Vector3d max_tc = mp_.map_max_boundary_ - camera_pt;
    Eigen::Vector3d min_tc = mp_.map_min_boundary_ - camera_pt;

    double min_t = 1000000;

    for (int i = 0; i < 3; ++i)
    {
        if (fabs(diff[i]) > 0)
        {

            double t1 = max_tc[i] / diff[i];
            if (t1 > 0 && t1 < min_t)
                min_t = t1;

            double t2 = min_tc[i] / diff[i];
            if (t2 > 0 && t2 < min_t)
                min_t = t2;
        }
    }

    return camera_pt + (min_t - 1e-3) * diff;
}

void GridMap::clearAndInflateLocalMap()
{
    /*clear outside local*/
    const int vec_margin = 5;
    // Eigen::Vector3i min_vec_margin = min_vec - Eigen::Vector3i(vec_margin,
    // vec_margin, vec_margin); Eigen::Vector3i max_vec_margin = max_vec +
    // Eigen::Vector3i(vec_margin, vec_margin, vec_margin);

    Eigen::Vector3i min_cut = md_.local_bound_min_ -
                              Eigen::Vector3i(mp_.local_map_margin_, mp_.local_map_margin_, mp_.local_map_margin_);
    Eigen::Vector3i max_cut = md_.local_bound_max_ +
                              Eigen::Vector3i(mp_.local_map_margin_, mp_.local_map_margin_, mp_.local_map_margin_);
    boundIndex(min_cut);
    boundIndex(max_cut);

    Eigen::Vector3i min_cut_m = min_cut - Eigen::Vector3i(vec_margin, vec_margin, vec_margin);
    Eigen::Vector3i max_cut_m = max_cut + Eigen::Vector3i(vec_margin, vec_margin, vec_margin);
    boundIndex(min_cut_m);
    boundIndex(max_cut_m);

    // clear data outside the local range

    for (int x = min_cut_m(0); x <= max_cut_m(0); ++x)
        for (int y = min_cut_m(1); y <= max_cut_m(1); ++y)
        {

            for (int z = min_cut_m(2); z < min_cut(2); ++z)
            {
                int idx = toAddress(x, y, z);
                md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
            }

            for (int z = max_cut(2) + 1; z <= max_cut_m(2); ++z)
            {
                int idx = toAddress(x, y, z);
                md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
            }
        }

    for (int z = min_cut_m(2); z <= max_cut_m(2); ++z)
        for (int x = min_cut_m(0); x <= max_cut_m(0); ++x)
        {

            for (int y = min_cut_m(1); y < min_cut(1); ++y)
            {
                int idx = toAddress(x, y, z);
                md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
            }

            for (int y = max_cut(1) + 1; y <= max_cut_m(1); ++y)
            {
                int idx = toAddress(x, y, z);
                md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
            }
        }

    for (int y = min_cut_m(1); y <= max_cut_m(1); ++y)
        for (int z = min_cut_m(2); z <= max_cut_m(2); ++z)
        {

            for (int x = min_cut_m(0); x < min_cut(0); ++x)
            {
                int idx = toAddress(x, y, z);
                md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
            }

            for (int x = max_cut(0) + 1; x <= max_cut_m(0); ++x)
            {
                int idx = toAddress(x, y, z);
                md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
            }
        }

    // inflate occupied voxels to compensate robot size

    int inf_step = ceil((mp_.obstacles_inflation_ - 0.001) / mp_.resolution_);
    // int inf_step_z = 1;
    std::vector<Eigen::Vector3i> inf_pts(pow(2 * inf_step + 1, 3));
    // inf_pts.resize(4 * inf_step + 3);
    Eigen::Vector3i inf_pt;

    // clear outdated data
    for (int x = md_.local_bound_min_(0); x <= md_.local_bound_max_(0); ++x)
        for (int y = md_.local_bound_min_(1); y <= md_.local_bound_max_(1); ++y)
            for (int z = md_.local_bound_min_(2); z <= md_.local_bound_max_(2); ++z)
            {
                md_.occupancy_buffer_inflate_[toAddress(x, y, z)] = 0;
            }

    // inflate obstacles
    for (int x = md_.local_bound_min_(0); x <= md_.local_bound_max_(0); ++x)
        for (int y = md_.local_bound_min_(1); y <= md_.local_bound_max_(1); ++y)
            for (int z = md_.local_bound_min_(2); z <= md_.local_bound_max_(2); ++z)
            {

                if (md_.occupancy_buffer_[toAddress(x, y, z)] > mp_.min_occupancy_log_)
                {
                    inflatePoint(Eigen::Vector3i(x, y, z), inf_step, inf_pts);

                    for (int k = 0; k < (int)inf_pts.size(); ++k)
                    {
                        inf_pt = inf_pts[k];
                        int idx_inf = toAddress(inf_pt);
                        if (idx_inf < 0 ||
                            idx_inf >= mp_.map_voxel_num_(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2))
                        {
                            continue;
                        }
                        md_.occupancy_buffer_inflate_[idx_inf] = 1;
                    }
                }
            }
}


void GridMap::updateOccupancy()
{
    if (toSecs(md_.last_occ_update_time_) < 1.0)
        md_.last_occ_update_time_ = getLatestStamp();

    if (!md_.occ_need_update_)
    {
        if (md_.flag_use_depth_fusion && (toSecs(getLatestStamp()) - toSecs(md_.last_occ_update_time_) > mp_.odom_depth_timeout_))
        {
              spdlog::error("odom or depth lost! ros::Time::now()={}, md_.last_occ_update_time_={}, mp_.odom_depth_timeout_={}",
                toSecs(getLatestStamp()), toSecs(md_.last_occ_update_time_), mp_.odom_depth_timeout_);
            md_.flag_depth_odom_timeout_ = true;
        }
        return;
    }
    md_.last_occ_update_time_ = getLatestStamp();

    /* update occupancy */
    // ros::Time t1, t2, t3, t4;
    // t1 = ros::Time::now();

    //   projectDepthImage();
    // should fill md.proj_points with points from frame and md.proj_points_cnt with num points

    // t2 = ros::Time::now();
    raycastProcess();
    // t3 = ros::Time::now();

    if (md_.local_updated_) {
        // std::cout << "local updated, clearing" << std::endl;
        update_map_mutex_.lock();
        clearAndInflateLocalMap();
        update_map_mutex_.unlock();
    }

    // t4 = ros::Time::now();

    // cout << setprecision(7);
    // cout << "t2=" << (t2-t1).toSec() << " t3=" << (t3-t2).toSec() << " t4=" << (t4-t3).toSec() << endl;;

    // md_.fuse_time_ += (t2 - t1).toSec();
    // md_.max_fuse_time_ = max(md_.max_fuse_time_, (t2 - t1).toSec());

    // if (mp_.show_occ_time_)
    //   ROS_WARN("Fusion: cur t = %lf, avg t = %lf, max t = %lf", (t2 - t1).toSec(),
    //            md_.fuse_time_ / md_.update_num_, md_.max_fuse_time_);

    md_.occ_need_update_ = false;
    md_.local_updated_ = false;
}

void GridMap::setCameraPosition(const Eigen::Vector3d &camera_pos)
{
    md_.camera_pos_ = camera_pos;
}

vkc::Shared<vkc::PointCloud> GridMap::convertToCapnpCloud(
    std::shared_ptr<open3d::geometry::PointCloud> cloud,
    vkc::Shared<vkc::Header> header)
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

        // uint8_t b = cloud->colors_[v].x() * 255;
        // uint8_t g = cloud->colors_[v].y() * 255;
        // uint8_t r = cloud->colors_[v].z() * 255;

        // Set the fields of the point
        *reinterpret_cast<float *>(pcData) = pt_x;
        *reinterpret_cast<float *>(pcData + 4) = pt_y;
        *reinterpret_cast<float *>(pcData + 8) = pt_z;

        *(pcData + 12) = 255;
        *(pcData + 13) = 255;
        *(pcData + 14) = 255;

        // Shift the pointer to the next point
        pcData += pointBytes;
    }

    // Truncate point cloud buffer to only store valid points
    // auto validBytes = static_cast<capnp::uint>(num_points * pointBytes);
    // orphan.truncate(validBytes);
    msg.setHeader(header.reader());
    msg.getHeader().setClockDomain(vkc::Header::ClockDomain::MONOTONIC);

    auto orphan = msg.disownPoints();
    msg.adoptPoints(kj::mv(orphan));

    return vkc::Shared<vkc::PointCloud>(std::move(mmb));
}

void GridMap::publishMap()
{
    auto latest_header = getLatestHeader();
    if (latest_header == nullptr)
    {
        spdlog::info("Waiting for odom");

        return;
    }
    // pcl::PointXYZ pt;
    // pcl::PointCloud<pcl::PointXYZ> cloud;
    std::shared_ptr<open3d::geometry::PointCloud> cloud = std::make_shared<open3d::geometry::PointCloud>();

    Eigen::Vector3i min_cut = md_.local_bound_min_;
    Eigen::Vector3i max_cut = md_.local_bound_max_;

    int lmm = mp_.local_map_margin_ / 2;
    min_cut -= Eigen::Vector3i(lmm, lmm, lmm);
    max_cut += Eigen::Vector3i(lmm, lmm, lmm);

    boundIndex(min_cut);
    boundIndex(max_cut);

    for (int x = min_cut(0); x <= max_cut(0); ++x)
        for (int y = min_cut(1); y <= max_cut(1); ++y)
            for (int z = min_cut(2); z <= max_cut(2); ++z)
            {
                if (md_.occupancy_buffer_[toAddress(x, y, z)] < mp_.min_occupancy_log_)
                    continue;

                Eigen::Vector3d pos;
                indexToPos(Eigen::Vector3i(x, y, z), pos);
                if (pos(2) > md_.camera_pos_.z() + mp_.visualization_truncate_height_ ||
                    pos(2) < md_.camera_pos_.z() - mp_.visualization_truncate_height_) {
                    continue;
                }
                cloud->points_.push_back(pos);
            }
    // spdlog::info("Publishing map with {} points", cloud->points_.size());
    auto capnp_cloud = this->convertToCapnpCloud(cloud, latest_header);
    this->map_receiver->handle(this->output_topic_, vkc::Message(capnp_cloud));
}

void GridMap::publishMapInflate(bool all_info)
{
    auto latest_header = getLatestHeader();
    if (latest_header == nullptr)
    {
        return;
    }

    // if (map_inf_pub_.getNumSubscribers() <= 0)
    //     return;

    // pcl::PointXYZ pt;
    // pcl::PointCloud<pcl::PointXYZ> cloud;
    std::shared_ptr<open3d::geometry::PointCloud> cloud = std::make_shared<open3d::geometry::PointCloud>();

    Eigen::Vector3i min_cut = md_.local_bound_min_;
    Eigen::Vector3i max_cut = md_.local_bound_max_;

    if (all_info)
    {
        int lmm = mp_.local_map_margin_;
        min_cut -= Eigen::Vector3i(lmm, lmm, lmm);
        max_cut += Eigen::Vector3i(lmm, lmm, lmm);
    }

    boundIndex(min_cut);
    boundIndex(max_cut);

    for (int x = min_cut(0); x <= max_cut(0); ++x)
        for (int y = min_cut(1); y <= max_cut(1); ++y)
            for (int z = min_cut(2); z <= max_cut(2); ++z)
            {
                if (md_.occupancy_buffer_inflate_[toAddress(x, y, z)] == 0)
                    continue;

                Eigen::Vector3d pos;
                indexToPos(Eigen::Vector3i(x, y, z), pos);
                if (pos(2) > md_.camera_pos_.z() + mp_.visualization_truncate_height_ ||
                    pos(2) < md_.camera_pos_.z() - mp_.visualization_truncate_height_) {
                    continue;
                }

                cloud->points_.push_back(pos);
            }
    // spdlog::info("Publishing inflated map with {} points", cloud->points_.size());
    auto capnp_cloud = this->convertToCapnpCloud(cloud, latest_header);
    this->map_inflate_receiver->handle(this->output_topic_ + "_inflate", vkc::Message(capnp_cloud));
}

bool GridMap::odomValid() { return md_.has_odom_; }

bool GridMap::hasDepthObservation() { return md_.has_first_depth_; }

Eigen::Vector3d GridMap::getOrigin() { return mp_.map_origin_; }

// int GridMap::getVoxelNum() {
//   return mp_.map_voxel_num_[0] * mp_.map_voxel_num_[1] * mp_.map_voxel_num_[2];
// }

void GridMap::getRegion(Eigen::Vector3d &ori, Eigen::Vector3d &size)
{
    ori = mp_.map_origin_, size = mp_.map_size_;
}

// void GridMap::depthOdomCallback(const sensor_msgs::ImageConstPtr &img,
//                                 const nav_msgs::OdometryConstPtr &odom)
// {
//     /* get pose */
//     Eigen::Quaterniond body_q = Eigen::Quaterniond(odom->pose.pose.orientation.w,
//                                                    odom->pose.pose.orientation.x,
//                                                    odom->pose.pose.orientation.y,
//                                                    odom->pose.pose.orientation.z);
//     Eigen::Matrix3d body_r_m = body_q.toRotationMatrix();
//     Eigen::Matrix4d body2world;
//     body2world.block<3, 3>(0, 0) = body_r_m;
//     body2world(0, 3) = odom->pose.pose.position.x;
//     body2world(1, 3) = odom->pose.pose.position.y;
//     body2world(2, 3) = odom->pose.pose.position.z;
//     body2world(3, 3) = 1.0;

//     Eigen::Matrix4d cam_T = body2world * md_.cam2body_;
//     md_.camera_pos_(0) = cam_T(0, 3);
//     md_.camera_pos_(1) = cam_T(1, 3);
//     md_.camera_pos_(2) = cam_T(2, 3);
//     md_.camera_r_m_ = cam_T.block<3, 3>(0, 0);

//     /* get depth image */
//     cv_bridge::CvImagePtr cv_ptr;
//     cv_ptr = cv_bridge::toCvCopy(img, img->encoding);
//     if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
//     {
//         (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, mp_.k_depth_scaling_factor_);
//     }
//     cv_ptr->image.copyTo(md_.depth_image_);

//     md_.occ_need_update_ = true;
//     md_.flag_use_depth_fusion = true;
// }

namespace cereal{
    template <class Archive>
    void serialize(Archive & ar, GridMapperParams &m){
        ar(
            cereal::make_nvp("tf_prefix", m.tf_prefix),
            cereal::make_nvp("resolution", m.resolution_),
            cereal::make_nvp("map_size_x", m.map_size_x_),
            cereal::make_nvp("map_size_y", m.map_size_y_),
            cereal::make_nvp("map_size_z", m.map_size_z_),
            cereal::make_nvp("local_update_range_x", m.local_update_range_x_),
            cereal::make_nvp("local_update_range_y", m.local_update_range_y_),
            cereal::make_nvp("local_update_range_z", m.local_update_range_z_),
            cereal::make_nvp("obstacles_inflation", m.obstacles_inflation_),
            cereal::make_nvp("p_hit", m.p_hit_),
            cereal::make_nvp("p_miss", m.p_miss_),
            cereal::make_nvp("p_min", m.p_min_),
            cereal::make_nvp("p_max", m.p_max_),
            cereal::make_nvp("p_occ", m.p_occ_),
            cereal::make_nvp("min_ray_length", m.min_ray_length_),
            cereal::make_nvp("max_ray_length", m.max_ray_length_),
            cereal::make_nvp("local_map_margin", m.local_map_margin_),
            cereal::make_nvp("device_version", m.device_version_),
            cereal::make_nvp("disparity_suffix", m.disparity_suffix_),
            cereal::make_nvp("input_cloud_topic", m.input_cloud_topic_),
            cereal::make_nvp("output_topic", m.output_topic_),
            cereal::make_nvp("odom_depth_timeout", m.odom_depth_timeout_),
            cereal::make_nvp("visualization_truncate_height", m.visualization_truncate_height_),
            cereal::make_nvp("skip_pixel", m.skip_pixel_));
}
}
;

struct StereoOdomData
{
    vkc::Shared<vkc::Image> image;
    vkc::Shared<vkc::Disparity> disparity;
    vkc::Shared<vkc::Odometry3d> odom;
};

struct CloudOdomData
{
    vkc::Shared<vkc::Odometry3d> odom;
    vkc::Shared<vkc::PointCloud> cloud;
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
using CloudSyncMap = tbb::concurrent_unordered_map<uint64_t, CloudOdomData>;
using ProcessUnit = tbb::concurrent_bounded_queue<StereoOdomData>;
using ProcessCloudUnit = tbb::concurrent_bounded_queue<CloudOdomData>;
using BatchSyncMap = tbb::concurrent_unordered_map<uint64_t, BatchStereoOdomData>;
using BatchProcessUnit = tbb::concurrent_bounded_queue<BatchStereoOdomData>;

struct Batch
{
    std::shared_ptr<BatchSyncMap> syncMap;
    std::shared_ptr<BatchProcessUnit> processUnit;
    int batch_size;
    // std::string trigger_topic;
};

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

void erase_before(std::shared_ptr<CloudSyncMap> &syncMap, uint64_t timestamp)
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

std::reference_wrapper<vkc::DataSource> GridMap::getVkcSource() {
    if (vkc_source_ref_ == nullopt)
        vkc_source_ref_ = std::ref(visualkit->source());
    return vkc_source_ref_.value();
}

void GridMap::runVkc() {
    std::cout << "Starting processing" << std::endl;
    visualkit->sink().start();
    getVkcSource().get().start();
    vkc::waitForCtrlCSignal();
    getVkcSource().get().stop();
    visualkit->sink().stop();
    std::cout << "Stopping processing" << std::endl;
}

bool left_updated = false;
bool right_updated = false;
class PointCloudGenerator
{

public:
    PointCloudGenerator(const std::string &direction, const vkc::PointCloudParams pcParams,
                        const GridMapperParams &genParams, std::shared_ptr<GridMap> gridMap)
        : direction(direction), pcParams(pcParams), genParams(genParams), gridMap(gridMap)
    {
        this->syncMap = std::make_shared<SyncMap>();
        this->cloudSyncMap = std::make_shared<CloudSyncMap>();
        this->processUnit = std::make_shared<ProcessUnit>();
        this->processCloudUnit = std::make_shared<ProcessCloudUnit>();
        this->processUnit->set_capacity(1);
        this->processCloudUnit->set_capacity(1);
        this->geomPtr = std::make_shared<open3d::geometry::PointCloud>();
        this->frameCount = 0;
        Eigen::Isometry3d T;
        T.linear() = Eigen::Matrix3d::Identity();
        T.translation() = Eigen::Vector3d{-gridMap->getOrigin()(0),
                                          -gridMap->getOrigin()(1),
                                          -gridMap->getOrigin()(2)};
        this->world_T_firstBody = T;
        
    }


    std::shared_ptr<CloudSyncMap> getCloudSyncMap()
    {
        return this->cloudSyncMap;
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

        spdlog::info("gen process data");
        while (true)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            this->frameCount++;
            if (!this->processUnit->try_pop(unit))
            {
                // spdlog::info("No data to process");
                continue;
            }

            auto disparityReader = unit.disparity.reader();
            auto imageReader = unit.image.reader();
            auto odomReader = unit.odom.reader();

            uint64_t lastModifiedCalib = std::max(imageReader.getIntrinsic().getLastModified(),
                                                  imageReader.getExtrinsic().getLastModified());

            this->body_T_camera = this->eCALSe3toEigen(
                imageReader.getExtrinsic().getBodyFrame());
            this->firstBody_T_latestBody = this->eCALSe3toEigen(odomReader.getPose());

            // set md_.cam2body_ as inverse of body_T_camera
            Eigen::Matrix3d cam2body_r_m = this->body_T_camera.linear().inverse();
            Eigen::Vector3d cam2body_t = -cam2body_r_m * this->body_T_camera.translation();
            Eigen::Matrix4d cam2body;
            cam2body.block<3, 3>(0, 0) = cam2body_r_m;
            cam2body(0, 3) = cam2body_t(0);
            cam2body(1, 3) = cam2body_t(1);
            cam2body(2, 3) = cam2body_t(2);
            gridMap->updateCam2Body(cam2body);
            

            // if (this->genParams.omit_left && direction == "left" || this->genParams.omit_right && direction == "right")
            // {
            //     continue;
            // }        
            // extracts point cloud from individual frames
            auto cloudPtr = std::make_shared<open3d::geometry::PointCloud>();
            this->generatePointCloud(cloudPtr, unit.disparity, unit.image);
            cloudPtr = this->preprocessPointCloud(cloudPtr, this->genParams);

            // no need depth_image since we creating here.
            // cv_ptr->image.copyTo(md_.depth_image_);

            Eigen::Affine3d pose = this->firstBody_T_latestBody * this->body_T_camera;

            gridMap->md_.camera_pos_(0) = pose.translation()(0);
            gridMap->md_.camera_pos_(1) = pose.translation()(1);
            gridMap->md_.camera_pos_(2) = pose.translation()(2);
            gridMap->md_.camera_r_m_ = pose.linear();

            // update map if shouldUpdateMap
            if (gridMap->shouldUpdateMap(gridMap->md_.camera_pos_))
            {
                gridMap->update_map_mutex_.lock();
                gridMap->updateMap(gridMap->md_.camera_pos_);
                spdlog::info("Map updated to {} {} {}", gridMap->mp_.map_origin_(0), gridMap->mp_.map_origin_(1), gridMap->mp_.map_origin_(2));
                gridMap->update_map_mutex_.unlock();
            }

            gridMap->md_.has_odom_ = true;
            gridMap->md_.update_num_ += 1;

            if (this->direction == "left")
            {
                left_updated = true;
            }
            if (this->direction == "right")
            {
                right_updated = true;
            }
            if (left_updated && right_updated)
            {
                gridMap->md_.occ_need_update_ = true;
                left_updated = false;
                right_updated = false;
            }
            // else
            // {   // camera always in map.
            //     gridMap->md_.occ_need_update_ = false;
            //     spdlog::info("Camera position is not in map {} {} {}", gridMap->md_.camera_pos_(0), gridMap->md_.camera_pos_(1), gridMap->md_.camera_pos_(2));
            // }

            gridMap->md_.flag_use_depth_fusion = true;

            gridMap->md_.proj_points_cnt = 0;
            gridMap->md_.proj_points_.clear();
            int i = 0;
            for (const auto &point : cloudPtr->points_)
            {
                i++;
                Eigen::Vector3d pt = point;
                gridMap->md_.proj_points_[gridMap->md_.proj_points_cnt++] = pt;
            }
        }
    }

    void processCloudData()
    {
        if (this->genParams.input_cloud_topic_.empty()) {
            spdlog::info("No input cloud");
            return;
        }
        spdlog::info("gen process cloud data");
        CloudOdomData unit;
        while (true)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            this->frameCount++;
            if (!this->processCloudUnit) {
                spdlog::error("processCloudUnit not defined");
                continue;
            }
            if (!this->processCloudUnit->try_pop(unit))
            {
                spdlog::info("No data to process");
                continue;
            }

            std::cout << "generating cloud from cloud "<< this->frameCount << std::endl;
            // extracts point cloud from individual frames
            auto cloudPtr = std::make_shared<open3d::geometry::PointCloud>();
            this->generatePointCloud(cloudPtr, unit.cloud, false);

            auto odomReader = unit.odom.reader();
            this->firstBody_T_latestBody = this->eCALSe3toEigen(odomReader.getPose());
            this->body_T_camera = Eigen::Affine3d::Identity();
            Eigen::Matrix3d cam2body_r_m = this->body_T_camera.linear().inverse();
            Eigen::Vector3d cam2body_t = -cam2body_r_m * this->body_T_camera.translation();
            Eigen::Matrix4d cam2body;
            cam2body.block<3, 3>(0, 0) = cam2body_r_m;
            cam2body(0, 3) = cam2body_t(0);
            cam2body(1, 3) = cam2body_t(1);
            cam2body(2, 3) = cam2body_t(2);
            gridMap->updateCam2Body(cam2body);

            Eigen::Affine3d pose = this->firstBody_T_latestBody * this->body_T_camera;

            gridMap->md_.camera_pos_(0) = pose.translation()(0);
            gridMap->md_.camera_pos_(1) = pose.translation()(1);
            gridMap->md_.camera_pos_(2) = pose.translation()(2);
            gridMap->md_.camera_r_m_ = pose.linear();
            // update map if shouldUpdateMap
            if (gridMap->shouldUpdateMap(gridMap->md_.camera_pos_))
            {
                gridMap->updateMap(gridMap->md_.camera_pos_);
                spdlog::info("Map updated to {} {} {}", gridMap->mp_.map_origin_(0), gridMap->mp_.map_origin_(1), gridMap->mp_.map_origin_(2));
            }
            if (!gridMap->isInMap(gridMap->md_.camera_pos_))
            {
                spdlog::error("Camera position is in map {} {} {}", gridMap->md_.camera_pos_(0), gridMap->md_.camera_pos_(1), gridMap->md_.camera_pos_(2));
            }
            gridMap->md_.has_odom_ = true;
            gridMap->md_.update_num_ += 1;
            gridMap->md_.occ_need_update_ = true;

            // gridMap->md_.flag_use_depth_fusion = false;
            gridMap->md_.proj_points_cnt = 0;
            gridMap->md_.proj_points_.clear();
            int i = 0;
            for (const auto &point : cloudPtr->points_)
            {
                i++;
                Eigen::Vector3d pt = point;
                gridMap->md_.proj_points_[gridMap->md_.proj_points_cnt++] = pt;
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

        cv::imwrite("image.png", imageMat);

        vkc::Shared<vkc::PointCloud> pointCloud = vkc::convertToPointCloud(
            disparity, imageMat.data, pcParams, gridMap->mp_.skip_pixel_);
        generatePointCloud(cloud, pointCloud, true);
    }

    void generatePointCloud(std::shared_ptr<open3d::geometry::PointCloud> cloud,
                            vkc::Shared<vkc::PointCloud> &pointCloud,
                            bool convertToWorld=true)
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
                world_pt_body = this->firstBody_T_latestBody * this->body_T_camera * point;
            }
            cloud->points_.push_back({world_pt_body.x(), world_pt_body.y(), world_pt_body.z()});
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
        std::shared_ptr<open3d::geometry::PointCloud> cloud, GridMapperParams genParams)
    {
        // if (genParams.preVoxelDownsample)
        // {
        //     std::cout << "Pre-voxel downsampling cloud for frame " << this->frameCount
        //               << " with " << std::to_string((int)cloud->points_.size()) << " points" << std::endl;
        //     cloud = cloud->VoxelDownSample(genParams.preVoxelSize);
        //     std::cout << "Post-voxel downsampling cloud for frame " << this->frameCount
        //               << " with " << std::to_string((int)cloud->points_.size()) << " points" << std::endl;
        // }
        return cloud;
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
    std::shared_ptr<CloudSyncMap> cloudSyncMap;
    std::shared_ptr<ProcessUnit> processUnit;
    std::shared_ptr<ProcessCloudUnit> processCloudUnit;

    uint64_t mLastModifiedCalib;
    Eigen::Affine3d body_T_camera;
    Eigen::Affine3d world_T_body;
    Eigen::Affine3d world_T_firstBody;
    Eigen::Affine3d firstBody_T_latestBody;
    Eigen::Affine3d map_T_firstBody;

    std::shared_ptr<open3d::geometry::PointCloud> geomPtr;
    int frameCount = 0;
    vkc::PointCloudParams pcParams;
    GridMapperParams genParams;
    std::shared_ptr<GridMap> gridMap;
    // std::shared_ptr<PointCloudAccumulator> accumulator;
    // std::shared_ptr<Batch> batch;
};

class PointCloudReceiver : public vkc::Receiver<vkc::PointCloud>
{

public:
    PointCloudReceiver(
        std::shared_ptr<CloudSyncMap> syncMap, std::shared_ptr<PointCloudGenerator> generator,
        std::shared_ptr<ProcessCloudUnit> processCloudUnit)
        : syncMap(syncMap), generator(generator), processCloudUnit(processCloudUnit) {}

    vkc::ReceiverStatus handle(const vkc::Message<vkc::Shared<vkc::PointCloud>> &message) override
    {
        vkc::Shared<vkc::PointCloud> cloud = message.payload;
        auto cloudReader = cloud.reader();
        auto cloudStamp = cloudReader.getHeader().getStampMonotonic() + cloudReader.getHeader().getClockOffset();

        auto it = this->syncMap->find(cloudStamp);
        if (it != this->syncMap->end())
        {
            it->second.cloud = cloud;
            if (it->second.odom != nullptr)
            {
                this->processCloudUnit->push(it->second);
                erase_before(this->syncMap, cloudStamp);
            }
        }
        else
        {
            CloudOdomData cloudOdom;
            cloudOdom.cloud = cloud;
            this->syncMap->emplace(cloudStamp, cloudOdom);
        }


        return vkc::ReceiverStatus::Open;
    }

private:
    std::shared_ptr<PointCloudGenerator> generator;
    std::shared_ptr<ProcessCloudUnit> processCloudUnit;
    std::shared_ptr<CloudSyncMap> syncMap;
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
    OdomReceiver(std::shared_ptr<SyncMap> syncMap,
                 std::shared_ptr<CloudSyncMap> cloudSyncMap,
                 std::shared_ptr<ProcessUnit> processUnit,
                 std::shared_ptr<ProcessCloudUnit> cloudProcessUnit,
                 std::shared_ptr<GridMap> gridMap)
        : syncMap(syncMap), cloudSyncMap(cloudSyncMap), processUnit(processUnit), gridMap(gridMap) {
            if (syncMap != nullptr && cloudSyncMap != nullptr)
            {
                spdlog::warn("Only one of the two should be non-null");
                cloudSyncMap = nullptr;
            }
            if (syncMap == nullptr && cloudSyncMap == nullptr)
            {
                spdlog::error("Both syncMap and cloudSyncMap are null");
            }
        }

    vkc::ReceiverStatus handle(const vkc::Message<vkc::Shared<vkc::Odometry3d>> &message) override
    {
        vkc::Shared<vkc::Odometry3d> odom = message.payload;
        auto odomReader = odom.reader();
        auto odomStamp = odomReader.getHeader().getStampMonotonic() + odomReader.getHeader().getClockOffset();

        Eigen::Vector3d position = {
            odomReader.getPose().getPosition().getX(),
            odomReader.getPose().getPosition().getY(),
            odomReader.getPose().getPosition().getZ()};
        gridMap->setCameraPosition(position);
        gridMap->setLatestHeader(vkc::Shared<vkc::Header>(odomReader.getHeader()));

        if (this->syncMap != nullptr) {
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
        }
        else
        {
            auto it = this->cloudSyncMap->find(odomStamp);
            if (it != this->cloudSyncMap->end())
            {
                it->second.odom = odom;

                if (it->second.cloud != nullptr)
                {
                    this->cloudProcessUnit->push(it->second);
                    erase_before(this->cloudSyncMap, odomStamp);
                }
            }
            else
            {
                CloudOdomData cloudOdom;
                cloudOdom.odom = odom;
                this->cloudSyncMap->emplace(odomStamp, cloudOdom);
            }
        }

        return vkc::ReceiverStatus::Open;
    }

private:
    std::shared_ptr<SyncMap> syncMap;
    std::shared_ptr<CloudSyncMap> cloudSyncMap;
    std::shared_ptr<ProcessUnit> processUnit;
    std::shared_ptr<ProcessCloudUnit> cloudProcessUnit;
    std::shared_ptr<GridMap> gridMap;

    double totalDistance;
    int counts;
};

int main(int argc, char *argv[])
{
    // ProgramArgs args;
    GridMapperParams genParams;

    std::string config_file;

    if (argc != 2) {
        spdlog::warn("Usage: ./grid_mapper <config_file>");
        return 0;
    } 
    config_file = std::string(argv[1]);

    std::ifstream ifs(config_file);
    if (ifs.is_open())
    {
        cereal::JSONInputArchive archive(ifs);
        archive(cereal::make_nvp("grid_mapper", genParams));
    }
    else
    {
        std::cout << "Could not open configuration file. Exiting..." << std::endl;
        return 0;
    }

    std::shared_ptr<GridMap> gridMap = std::make_shared<GridMap>();
    gridMap->initMap(genParams);

    auto sourceRef = gridMap->getVkcSource();
    std::vector<std::string> topics;

    std::cout << genParams.disparity_suffix_ << std::endl;

    std::vector<std::pair<std::string, std::string>> imageAndDisparityTopics;
    std::pair<std::string, std::string> leftStereoPairS0;
    std::pair<std::string, std::string> rightStereoPairS0;
    std::pair<std::string, std::string> leftStereoPairS1;
    std::pair<std::string, std::string> rightStereoPairS1;
    std::string S0OdomTopic, S1OdomTopic;

    leftStereoPairS0.first = "S0/stereo1_l";
    rightStereoPairS0.first = "S0/stereo2_r";
    leftStereoPairS0.second = "S0/stereo1_l/" + genParams.disparity_suffix_;
    rightStereoPairS0.second = "S0/stereo2_r/" + genParams.disparity_suffix_;
    leftStereoPairS1.first = "S1/stereo1_l";
    rightStereoPairS1.first = "S1/stereo2_r";
    leftStereoPairS1.second = "S1/stereo1_l/" + genParams.disparity_suffix_;
    rightStereoPairS1.second = "S1/stereo2_r/" + genParams.disparity_suffix_;
    S0OdomTopic = "S0/vio_odom";
    S1OdomTopic = "S1/vio_odom";

    vkc::PointCloudParams pcParams;
    pcParams.disparityOffset = 1;
    pcParams.maxDepth = 5.0;


    if (genParams.input_cloud_topic_ != "")
    {
        auto direction = "cloud";
        //create single generator for pointcloud

        auto generator = std::make_shared<PointCloudGenerator>(
            direction, pcParams, genParams, gridMap);

        std::shared_ptr<CloudSyncMap> cloudSyncMap = generator->getCloudSyncMap();
        std::shared_ptr<ProcessCloudUnit> cloudProcessUnit = generator->getProcessCloudUnit();

        auto cloudReceiver = std::make_unique<PointCloudReceiver>(cloudSyncMap, generator, cloudProcessUnit);
        sourceRef.get().install(genParams.input_cloud_topic_, std::move(cloudReceiver));

        std::cout << "Installing receivers for odometry" << std::endl;
        auto odomReceiver = std::make_unique<OdomReceiver>(nullptr, cloudSyncMap, nullptr, cloudProcessUnit, gridMap);
        // since cloud has no odom, just use odom of either stereo pair
        sourceRef.get().install(S0OdomTopic, std::move(odomReceiver));

        std::thread t([generator]()
        { generator->processCloudData(); });
        t.detach();
    }
    else
    {
        if (genParams.device_version_ == "180-P" || genParams.device_version_ == "360")
        {
            imageAndDisparityTopics.push_back(leftStereoPairS1);
            imageAndDisparityTopics.push_back(rightStereoPairS1);
        }

        if (genParams.device_version_ != "180-P")
        {
            imageAndDisparityTopics.push_back(leftStereoPairS0);
            imageAndDisparityTopics.push_back(rightStereoPairS0);
        }

        std::string direction; // either left or right

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

            auto generator = std::make_shared<PointCloudGenerator>(
                direction, pcParams,
                genParams, gridMap);

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
            auto odomReceiver = std::make_unique<OdomReceiver>(syncMap, nullptr, processUnit, nullptr, gridMap);
            sourceRef.get().install(odomTopic, std::move(odomReceiver));

            std::thread t([generator]()
                        { generator->processData(); });
            t.detach();
        }
    }

    auto updateOccupancyThread = std::thread([&]() {
        while (true)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            gridMap->updateOccupancy();
        }
    });

    auto fadingThread = std::thread([&]() {
        while (true)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            gridMap->fade();
        }
    });

    auto visCallbackThread = std::thread([&]() {
        int i = 0;
        while (true) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            gridMap->publishMapInflate(true);
            gridMap->publishMap();
            i++;
            if (i % 100 == 0) {
                spdlog::info("Publishing map");
            }
        }
    });
    gridMap->runVkc();

    updateOccupancyThread.join();
    visCallbackThread.join();
    fadingThread.join();

    return 0;
}