#ifndef VKC_SDK_SDK_HPP
#define VKC_SDK_SDK_HPP

#include "capnp/system.capnp.h"
#define VKC_SDK_VERSION "1.4.0"

#include <vk_sdk/Logging.hpp>
#include <vk_sdk/Message.hpp>
#include <vk_sdk/Receivers.hpp>
#include <vk_sdk/Data.hpp>
#include <vk_sdk/VisualKit.hpp>
#include <vk_sdk/Utilities.hpp>

namespace vkc {
    /// Convenience function to connect a data source to a data sink for a given topic.
    ///
    /// This function can optionally be parameterized with an additional `TRANSCEIVER` template argument.
    /// The `TRANSCEIVER` argument itself must be a templatable class that contains a `wrap` method that
    /// have the signature of `std::unique_ptr<Receiver<T>> wrap(std::unique_ptr<Receiver<T>>);`. The 
    /// `TRANSCEIVER` is used to map the original Receiver obtained from the sink into a new Receiver
    /// before it is installed into the source. (See `vkc::IdentityReceiver` for an example of how to 
    /// declare such a class.)
    ///
    /// This function returns the installed receiver's `ReceiverId` if successful. Otherwise,
    /// it returns an invalid `ReceiverId`. 
    template<typename T, template<typename> class TRANSCEIVER = vkc::IdentityReceiver>
    inline ReceiverId connectReceiver(std::string_view topic, DataSource &source, DataSink &sink) {
        auto originalReceiver = sink.obtain(topic, Type<T>());
        if (originalReceiver == nullptr) return ReceiverId::invalid();
        auto wrappingReceiver = TRANSCEIVER<T>::wrap(std::move(originalReceiver));
        if (wrappingReceiver == nullptr) return ReceiverId::invalid();
        return source.install(topic, std::move(wrappingReceiver));
    }

    /// Convenience function to connect a data source to a data sink for a known VisualKit topic.
    ///
    /// This function differs from `connectReceiver` in that a type parameter need not be passed into this
    /// function since VisualKit topic types are always known. For information about the `TRANSCEIVER` type 
    /// parameter, see the documentation for `connectReceiver` since they are equivalent in purpose.
    ///
    /// This function returns the installed receiver's `ReceiverId` if successful. Otherwise,
    /// it returns an invalid `ReceiverId`.
    template<template<typename> class TRANSCEIVER = vkc::IdentityReceiver>
    inline ReceiverId connectVisualKitReceiver(std::string_view topic, DataSource &source, DataSink &sink) {
        if (topic == "ws/message") return connectReceiver<ManagerMessage, TRANSCEIVER>(topic, source, sink);

        if (topic == "S0/imu") return connectReceiver<Imu, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/imu") return connectReceiver<Imu, TRANSCEIVER>(topic, source, sink);

        if (topic == "S0/gps") return connectReceiver<GPS, TRANSCEIVER>(topic, source, sink);
        
        if (topic == "S0/imu_list") return connectReceiver<ImuList, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/imu_list") return connectReceiver<ImuList, TRANSCEIVER>(topic, source, sink);

        if (topic == "S0/cama") return connectReceiver<Image, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/camb") return connectReceiver<Image, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/camc") return connectReceiver<Image, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/camd") return connectReceiver<Image, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/cama") return connectReceiver<Image, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/camb") return connectReceiver<Image, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/camc") return connectReceiver<Image, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/camd") return connectReceiver<Image, TRANSCEIVER>(topic, source, sink);

        if (topic == "S0/stereo1_l") return connectReceiver<Image, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/stereo1_r") return connectReceiver<Image, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/stereo2_l") return connectReceiver<Image, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/stereo2_r") return connectReceiver<Image, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/stereo1_l") return connectReceiver<Image, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/stereo1_r") return connectReceiver<Image, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/stereo2_l") return connectReceiver<Image, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/stereo2_r") return connectReceiver<Image, TRANSCEIVER>(topic, source, sink);

        if (topic == "S0/cama/tags") return connectReceiver<TagDetections, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/camb/tags") return connectReceiver<TagDetections, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/camc/tags") return connectReceiver<TagDetections, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/camd/tags") return connectReceiver<TagDetections, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/cama/tags") return connectReceiver<TagDetections, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/camb/tags") return connectReceiver<TagDetections, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/camc/tags") return connectReceiver<TagDetections, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/camd/tags") return connectReceiver<TagDetections, TRANSCEIVER>(topic, source, sink);

        if (topic == "S0/cama/yolo") return connectReceiver<Detections2d, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/camb/yolo") return connectReceiver<Detections2d, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/camc/yolo") return connectReceiver<Detections2d, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/camd/yolo") return connectReceiver<Detections2d, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/cama/yolo") return connectReceiver<Detections2d, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/camb/yolo") return connectReceiver<Detections2d, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/camc/yolo") return connectReceiver<Detections2d, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/camd/yolo") return connectReceiver<Detections2d, TRANSCEIVER>(topic, source, sink);

        if (topic == "S0/stereo1_l/disparity") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/stereo1_r/disparity") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/stereo2_l/disparity") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/stereo2_r/disparity") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/stereo1_l/disparity") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/stereo1_r/disparity") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/stereo2_l/disparity") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/stereo2_r/disparity") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);

        if (topic == "S0/stereo1_l/confidence_map") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/stereo1_r/confidence_map") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/stereo2_l/confidence_map") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/stereo2_r/confidence_map") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/stereo1_l/confidence_map") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/stereo1_r/confidence_map") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/stereo2_l/confidence_map") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/stereo2_r/confidence_map") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);

        if (topic == "S0/stereo1_l/filtered/disparity") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/stereo1_r/filtered/disparity") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/stereo2_l/filtered/disparity") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/stereo2_r/filtered/disparity") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/stereo1_l/filtered/disparity") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/stereo1_r/filtered/disparity") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/stereo2_l/filtered/disparity") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/stereo2_r/filtered/disparity") return connectReceiver<Disparity, TRANSCEIVER>(topic, source, sink);

        if (topic == "S0/cama/hfflow") return connectReceiver<HFOpticalFlowResult, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/camb/hfflow") return connectReceiver<HFOpticalFlowResult, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/camc/hfflow") return connectReceiver<HFOpticalFlowResult, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/camd/hfflow") return connectReceiver<HFOpticalFlowResult, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/cama/hfflow") return connectReceiver<HFOpticalFlowResult, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/camb/hfflow") return connectReceiver<HFOpticalFlowResult, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/camc/hfflow") return connectReceiver<HFOpticalFlowResult, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/camd/hfflow") return connectReceiver<HFOpticalFlowResult, TRANSCEIVER>(topic, source, sink);

        if (topic == "S0/stereo1_l/hfflow") return connectReceiver<HFOpticalFlowResult, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/stereo1_r/hfflow") return connectReceiver<HFOpticalFlowResult, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/stereo2_l/hfflow") return connectReceiver<HFOpticalFlowResult, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/stereo2_r/hfflow") return connectReceiver<HFOpticalFlowResult, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/stereo1_l/hfflow") return connectReceiver<HFOpticalFlowResult, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/stereo1_r/hfflow") return connectReceiver<HFOpticalFlowResult, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/stereo2_l/hfflow") return connectReceiver<HFOpticalFlowResult, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/stereo2_r/hfflow") return connectReceiver<HFOpticalFlowResult, TRANSCEIVER>(topic, source, sink);

        if (topic == "S0/vio_odom") return connectReceiver<Odometry3d, TRANSCEIVER>(topic, source, sink);
        if (topic == "S0/vio_odom_ned") return connectReceiver<Odometry3d, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/vio_odom") return connectReceiver<Odometry3d, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/vio_odom_ned") return connectReceiver<Odometry3d, TRANSCEIVER>(topic, source, sink);

        if (topic == "S0/vio_state") return connectReceiver<VioState, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/vio_state") return connectReceiver<VioState, TRANSCEIVER>(topic, source, sink);

        if (topic == "S0/panorama") return connectReceiver<Image, TRANSCEIVER>(topic, source, sink);
        if (topic == "S1/panorama") return connectReceiver<Image, TRANSCEIVER>(topic, source, sink);

        return ReceiverId::invalid();
    }
}

#endif