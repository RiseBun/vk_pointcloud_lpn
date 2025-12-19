#ifndef VKC_SDK_RECEIVERS_HPP
#define VKC_SDK_RECEIVERS_HPP

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <thread>
#include <sstream>
#include <tuple>

#include "vk_sdk/Logging.hpp"
#include "vk_sdk/Message.hpp"
#include "vk_sdk/capnp/Shared.hpp"

constexpr int QUEUE_WARN_SIZE = 100;

namespace vkc {

    /// Status returned by a Receiver about itself when it handles a message.
    enum class ReceiverStatus {
        Open,       //< Receiver is still handling messages.
        Closed,     //< Receiver is no longer handling messages.
    };

    /// Identifier for a receiver after it has been installed into a `DataSource`.
    ///
    /// If installation fails, the returned identifier will be an invalid identifier.
    class ReceiverId {
    private:
        uint64_t mValue;

    public:
        /// Returns an invalid identifier.
        ///
        /// This is equivalent to calling the default constructor of this class.
        static constexpr ReceiverId invalid() { return ReceiverId(0); }

        /// Default constructor that constructs an invalid receiver identifier.
        constexpr ReceiverId(): mValue(0) {}
        /// Trivial constructor that constructs a receiver identifier from a 64-bit integer.
        constexpr ReceiverId(uint64_t value): mValue(value) {}

        /// Returns whether the identifier is valid.
        constexpr bool isValid() { return mValue != 0; }
        /// Returns whether the identifier is invalid.
        constexpr bool isInvalid() { return mValue == 0; }

        constexpr bool operator<(ReceiverId rhs) const { return mValue < rhs.mValue; }
        constexpr bool operator>(ReceiverId rhs) const { return mValue > rhs.mValue; }
        constexpr bool operator<=(ReceiverId rhs) const { return mValue <= rhs.mValue; }
        constexpr bool operator>=(ReceiverId rhs) const { return mValue >= rhs.mValue; }
        constexpr bool operator==(ReceiverId rhs) const { return mValue == rhs.mValue; }
        constexpr bool operator!=(ReceiverId rhs) const { return mValue != rhs.mValue; }
    };

    /// Receiver callback for a message.
    ///
    /// Note: 
    /// 1) Receiver methods may be called from a different thread than the thread that the receiver is created in.
    /// 2) However, receivers are not thread-safe and must not be invoked simultaneously from different threads.
    template<typename T>
    struct Receiver {
        virtual ~Receiver() = default;
        /// (Note: Superceded by overloaded variant which takes in topic name as well (since v1.3.0).)
        ///
        /// Receive the incoming message and handle it. 
        ///
        /// Implementations must override this method to handle a message being received.
        ///
        /// This method should not take too much time so as to not block the data source that invoked this
        /// method. Blocking the data source might cause subsequent messages (possibly from an irrelevant
        /// topic) to be delayed or dropped depending on the data source's implementation.
        ///
        /// If you have computationally heavy task, consider offloading it to a separate thread/threadpool.
        ///
        /// Implementation should return a `ReceiverStatus` value as a hint to the data source whether
        /// further messages would still be handled. If `ReceiverStatus::Closed` is returned, a data
        /// source may choose to perform early cleanup and cease further sending of messages to this
        /// receiver.
        virtual ReceiverStatus handle(const Message<Shared<T>>& message) {
            return this->handle("<Unknown>", message);
        }

        /// Receive the incoming message and handle it.
        ///
        /// Implementations must override this method to handle a message being received.
        ///
        /// This method should not take too much time so as to not block the data source that invoked this
        /// method. Blocking the data source might cause subsequent messages (possibly from an irrelevant
        /// topic) to be delayed or dropped depending on the data source's implementation.
        ///
        /// If you have computationally heavy task, consider offloading it to a separate thread/threadpool.
        ///
        /// Implementation should return a `ReceiverStatus` value as a hint to the data source whether
        /// further messages would still be handled. If `ReceiverStatus::Closed` is returned, a data
        /// source may choose to perform early cleanup and cease further sending of messages to this
        /// receiver.
        ///
        /// Note: This method was introduced in version 1.3.0 of the SDK.
        /// - For newly implemented senders, this should be the preferred method to call over the original 
        /// `handle` method.
        /// - For newly implemented receivers, this should be the preferred method to implement.
        virtual ReceiverStatus handle(const std::string_view sourceTopic, const Message<Shared<T>>& message) {
            std::ignore = sourceTopic; // Suppress unused warning.
            return this->handle(message);
        }
    };

    /// A Receiver that wraps another Receiver and invokes the wrapped Receiver asynchronously in
    /// a dedicated thread.
    template <typename T>
    class AsyncThreadedReceiver: public Receiver<T> {
    private:
        struct State {
            std::mutex mMutex;
            std::condition_variable mCondVar;
            std::optional<std::deque<Message<Shared<T>>>> mQueue = std::deque<Message<Shared<T>>>();
            std::string mTopic;
            bool mTerminated = false;
            size_t mHighWatermark = 10;
        };

        std::shared_ptr<State> mState;

        explicit AsyncThreadedReceiver(std::shared_ptr<State> state) : mState(std::move(state)) {}

    public:
        /// Wraps the given receiver in an `AsyncThreadedReceiver` and returns it.
        static std::unique_ptr<Receiver<T>> wrap(std::unique_ptr<Receiver<T>> inner) {
            std::shared_ptr<State> state(new State());
            std::shared_ptr<Receiver<T>> receiver = std::move(inner);

            std::thread([state, receiver = std::move(receiver)]() {
                std::string topic;

                while (true) {
                    std::unique_lock lock(state->mMutex);

                    while (state->mQueue.value().empty()) {
                        if (state->mTerminated) {
                            return;
                        }
                        state->mCondVar.wait(lock);
                    }

                    if (topic.empty()) {
                        topic = state->mTopic;
                    }

                    Message<Shared<T>> message = state->mQueue.value().front();
                    state->mQueue.value().pop_front();

                    auto queueSize = state->mQueue.value().size();
                    auto highWatermark = state->mHighWatermark;
                    auto shrunkSignificantly = (queueSize > (QUEUE_WARN_SIZE / 2)) && (queueSize <= highWatermark / 2);

                    // we do not want mHighWatermark become too small, smaller than half the warning size
                    if (shrunkSignificantly) {
                        state->mHighWatermark = queueSize;
                    }

                    lock.unlock();

                    if (shrunkSignificantly) {
                        std::ostringstream oss;
                        oss << "Internal queue for topic '" << topic << "' has shrunk 50% in size (now " << queueSize << ").";
                        vkc::log(LogLevel::INFO, oss.str());
                    }

                    if (receiver->handle(message) == ReceiverStatus::Closed) {
                        break;
                    }
                }

                std::lock_guard lock(state->mMutex);
                state->mQueue = std::nullopt;
            }).detach();

            return std::unique_ptr<AsyncThreadedReceiver<T>>(new AsyncThreadedReceiver<T>(state));
        }

        ~AsyncThreadedReceiver() {
            {
                std::lock_guard lock(mState->mMutex);
                mState->mTerminated = true;
            }
            mState->mCondVar.notify_one();
        }

        ReceiverStatus handle(const Message<Shared<T>>& message) override {
            return this->handle("<Unknown>", message);
        }

        ReceiverStatus handle(const std::string_view topic, const Message<Shared<T>>& message) override {
            std::unique_lock lock(mState->mMutex);
            if (mState->mQueue.has_value()) {
                if (mState->mTopic.empty()) {
                    mState->mTopic = topic;
                }

                mState->mQueue.value().push_back(message);

                auto queueSize = mState->mQueue.value().size();
                auto highWatermark = mState->mHighWatermark;
                auto grownSignificantly = queueSize >= highWatermark * 2;
                if (grownSignificantly) {
                    mState->mHighWatermark = queueSize;
                }

                lock.unlock();
                mState->mCondVar.notify_one();

                if (grownSignificantly) {
                    std::ostringstream oss;
                    oss << "Internal queue for topic '" << topic << "' has grown double in size (now " << queueSize << ").";
                    vkc::log(LogLevel::WARN, oss.str());
                }

                return ReceiverStatus::Open;
            } else {
                return ReceiverStatus::Closed;
            }
        }
    };

    /// Static class used only to map a receiver back to itself.
    ///
    /// This is the default transceiver used by the `vkc::connectReceiver` function.
    template <typename T>
    class IdentityReceiver: public Receiver<T> {
    public:
        static std::unique_ptr<Receiver<T>> wrap(std::unique_ptr<Receiver<T>> receiver) {
            return receiver;
        }

    private:
        IdentityReceiver() = default;
    };

}


#endif