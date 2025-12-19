#ifndef VKC_SDK_DATA_HPP
#define VKC_SDK_DATA_HPP

#include <memory>
#include <string_view>

#include <vk_sdk/capnp/disparity.capnp.h>
#include <vk_sdk/capnp/image.capnp.h>
#include <vk_sdk/capnp/imu.capnp.h>
#include <vk_sdk/capnp/imulist.capnp.h>
#include <vk_sdk/capnp/gps.capnp.h>
#include <vk_sdk/capnp/tagdetection.capnp.h>
#include <vk_sdk/capnp/odometry3d.capnp.h>
#include <vk_sdk/capnp/flow2d.capnp.h>
#include <vk_sdk/capnp/pointcloud.capnp.h>
#include <vk_sdk/capnp/system.capnp.h>
#include <vk_sdk/capnp/detection2d.capnp.h>
#include <vk_sdk/capnp/cameracontrol.capnp.h>
#include <vk_sdk/capnp/mavstate.capnp.h>

#include "vk_sdk/Receivers.hpp"
#include "vk_sdk/Utilities.hpp"

namespace vkc {

    /// Represents a source of data.
    class DataSource {
    public:
        /// Destroy all resources associated with the data source.
        ///
        /// This method will stop the data source without waiting for data exhaustion.
        virtual ~DataSource() {};

        /// Start forwarding messages from this data source to its receivers.
        virtual void start() = 0;

        /// Stop forwarding messages from this data source to its receivers.
        ///
        /// If the argument `waitForExhaustion` is `true`, this method will block until all data
        /// from the data source is exhausted before stopping. Some data sources will never exhaust and
        /// thus calling stop on these sources will block indefinitely.
        virtual void stop(bool waitForExhaustion = false) = 0;

        /// Install a receiver for the given topic in the data source.
        ///
        /// This method returns an invalid ID if the data source does not support reading from the given topic or type.
        virtual ReceiverId install(std::string_view topic, std::unique_ptr<Receiver<ManagerMessage>> receiver) = 0;
        /// Install a receiver for the given topic in the data source.
        ///
        /// This method returns an invalid ID if the data source does not support reading from the given topic or type.
        virtual ReceiverId install(std::string_view topic, std::unique_ptr<Receiver<Image>> receiver) = 0;
        /// Install a receiver for the given topic in the data source.
        ///
        /// This method returns an invalid ID if the data source does not support reading from the given topic or type.
        virtual ReceiverId install(std::string_view topic, std::unique_ptr<Receiver<Imu>> receiver) = 0;
        /// Install a receiver for the given topic in the data source.
        ///
        /// This method returns an invalid ID if the data source does not support reading from the given topic or type.
        virtual ReceiverId install(std::string_view topic, std::unique_ptr<Receiver<ImuList>> receiver) = 0;
        /// Install a receiver for the given topic in the data source.
        ///
        /// This method returns an invalid ID if the data source does not support reading from the given topic or type.
        virtual ReceiverId install(std::string_view topic, std::unique_ptr<Receiver<GPS>> receiver) = 0;
        /// Install a receiver for the given topic in the data source.
        ///
        /// This method returns an invalid ID if the data source does not support reading from the given topic or type.
        virtual ReceiverId install(std::string_view topic, std::unique_ptr<Receiver<TagDetections>> receiver) = 0;
        /// Install a receiver for the given topic in the data source.
        ///
        /// This method returns an invalid ID if the data source does not support reading from the given topic or type.
        virtual ReceiverId install(std::string_view topic, std::unique_ptr<Receiver<Disparity>> receiver) = 0;
        /// Install a receiver for the given topic in the data source.
        ///
        /// This method returns an invalid ID if the data source does not support reading from the given topic or type.
        virtual ReceiverId install(std::string_view topic, std::unique_ptr<Receiver<Odometry3d>> receiver) = 0;
        /// Install a receiver for the given topic in the data source.
        ///
        /// This method returns an invalid ID if the data source does not support reading from the given topic or type.
        virtual ReceiverId install(std::string_view topic, std::unique_ptr<Receiver<HFOpticalFlowResult>> receiver) = 0;
        /// Install a receiver for the given topic in the data source.
        ///
        /// This method returns an invalid ID if the data source does not support reading from the given topic or type.
        virtual ReceiverId install(std::string_view topic, std::unique_ptr<Receiver<VioState>> receiver) = 0;
        /// Install a receiver for the given topic in the data source.
        ///
        /// This method returns an invalid ID if the data source does not support reading from the given topic or type.
        virtual ReceiverId install(std::string_view topic, std::unique_ptr<Receiver<Detections2d>> receiver) = 0;        
        /// Install a receiver for the given topic in the data source.
        ///
        /// This method returns an invalid ID if the data source does not support reading from the given topic or type.
        virtual ReceiverId install(std::string_view topic, std::unique_ptr<Receiver<PointCloud>> receiver) = 0;
        /// Install a receiver for the given topic in the data source.
        ///
        /// This method returns an invalid ID if the data source does not support reading from the given topic or type.
        virtual ReceiverId install(std::string_view topic, std::unique_ptr<Receiver<CameraControl>> receiver) = 0;        
        /// Install a receiver for the given topic in the data source.
        ///
        /// This method returns an invalid ID if the data source does not support reading from the given topic or type.
        virtual ReceiverId install(std::string_view topic, std::unique_ptr<Receiver<MavState>> receiver) = 0;
        /// Remove and destroy a receiver that had been installed into the data source.
        ///
        /// The receiver will no longer receive any message after calling this method.
        virtual void remove(ReceiverId id) = 0;

        /// Remove and destroy all receivers that had been installed into the data source.
        virtual void clear() = 0;
        
    };

    /// Represents a destination of data.
    class DataSink {
    public:
        /// Destroy all resources associated with the data sink.
        ///
        /// This method will stop the data sink without waiting for data exhaustion.
        virtual ~DataSink() {};

        /// Start accepting messages into this data sink from its receivers.
        virtual void start() = 0;

        /// Stop accepting messages into this data sink from its receivers.
        ///
        /// If the argument `waitForExhaustion` is `true`, this method blocks until there can be no more
        /// incoming messages from the receivers handed out by it (data exhaustion). Since a caller can always invoke 
        /// the `Receiver::handle` method at any time, a data sink defines data exhaustion as only when ALL receivers 
        /// that has been handed out by it has been destroyed.
        virtual void stop(bool waitForExhaustion = false) = 0;

        /// Obtain a receiver for the given topic in the data sink.
        ///
        /// This method returns null if the data sink does not support writing to the given topic or type.
        virtual std::unique_ptr<Receiver<ManagerMessage>> obtain(std::string_view topic, Type<ManagerMessage> type) = 0;
        /// Obtain a receiver for the given topic in the data sink.
        ///
        /// This method returns null if the data sink does not support writing to the given topic or type.
        virtual std::unique_ptr<Receiver<ManagerCommand>> obtain(std::string_view topic, Type<ManagerCommand> type) = 0;
        /// Obtain a receiver for the given topic in the data sink.
        ///
        /// This method returns null if the data sink does not support writing to the topic or type.
        virtual std::unique_ptr<Receiver<Image>> obtain(std::string_view topic, Type<Image> type) = 0;
        /// Obtain a receiver for the given topic in the data sink.
        ///
        /// This method returns null if the data sink does not support writing to the topic or type.
        virtual std::unique_ptr<Receiver<Imu>> obtain(std::string_view topic, Type<Imu> type) = 0;
        /// Obtain a receiver for the given topic in the data sink.
        ///
        /// This method returns null if the data sink does not support writing to the topic or type.
        virtual std::unique_ptr<Receiver<ImuList>> obtain(std::string_view topic, Type<ImuList> type) = 0;
        /// Obtain a receiver for the given topic in the data sink.
        ///
        /// This method returns null if the data sink does not support writing to the topic or type.
        virtual std::unique_ptr<Receiver<GPS>> obtain(std::string_view topic, Type<GPS> type) = 0;
        /// Obtain a receiver for the given topic in the data sink.
        ///
        /// This method returns null if the data sink does not support writing to the topic or type.
        virtual std::unique_ptr<Receiver<TagDetections>> obtain(std::string_view topic, Type<TagDetections> type) = 0;
        /// Obtain a receiver for the given topic in the data sink.
        ///
        /// This method returns null if the data sink does not support writing to the topic or type.
        virtual std::unique_ptr<Receiver<Disparity>> obtain(std::string_view topic, Type<Disparity> type) = 0;
        /// Obtain a receiver for the given topic in the data sink.
        ///
        /// This method returns null if the data sink does not support writing to the topic or type.
        virtual std::unique_ptr<Receiver<Odometry3d>> obtain(std::string_view topic, Type<Odometry3d> type) = 0;
        /// Obtain a receiver for the given topic in the data sink.
        ///
        /// This method returns null if the data sink does not support writing to the topic or type.
        virtual std::unique_ptr<Receiver<HFOpticalFlowResult>> obtain(std::string_view topic, Type<HFOpticalFlowResult> type) = 0;
        /// Obtain a receiver for the given topic in the data sink.
        ///
        /// This method returns null if the data sink does not support writing to the topic or type.
        virtual std::unique_ptr<Receiver<VioState>> obtain(std::string_view topic, Type<VioState> type) = 0;
        /// Obtain a receiver for the given topic in the data sink.
        ///
        /// This method returns null if the data sink does not support writing to the topic or type.
        virtual std::unique_ptr<Receiver<Detections2d>> obtain(std::string_view topic, Type<Detections2d> type) = 0;
        /// Obtain a receiver for the given topic in the data sink.
        ///
        /// This method returns null if the data sink does not support writing to the topic or type.
        virtual std::unique_ptr<Receiver<PointCloud>> obtain(std::string_view topic, Type<PointCloud> type) = 0;
        /// Obtain a receiver for the given topic in the data sink.
        ///
        /// This method returns null if the data sink does not support writing to the topic or type.
        virtual std::unique_ptr<Receiver<CameraControl>> obtain(std::string_view topic, Type<CameraControl> type) = 0;
        /// Obtain a receiver for the given topic in the data sink.
        ///
        /// This method returns null if the data sink does not support writing to the topic or type.
        virtual std::unique_ptr<Receiver<MavState>> obtain(std::string_view topic, Type<MavState> type) = 0;


    };
}

#endif