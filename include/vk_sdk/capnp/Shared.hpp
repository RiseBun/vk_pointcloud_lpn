#ifndef VKC_OWNED_HPP
#define VKC_OWNED_HPP

#include <kj/io.h>
#include <memory>
#include <capnp/message.h>
#include <capnp/serialize.h>

namespace vkc {
    /// Shared ownership to a capnproto value.
    template <typename T>
    class Shared {
    public:
        Shared() : mBuffer(nullptr) {};

        /// Construct an owned version of a capnproto type `T` by consuming a raw builder.
        explicit Shared(std::unique_ptr<capnp::MessageBuilder> raw) : Shared() {
            mBuffer = std::move(raw);
            mPointer = mBuffer->template getRoot<T>().asReader();
        }

        bool empty() {return mBuffer == nullptr;}

        /// Construct an owned version of a capnproto type `T` by deserializing (with validation and copy)
        /// from the stream.
        explicit Shared(kj::InputStream& stream): Shared() {
            auto buffer = new capnp::MallocMessageBuilder();
            capnp::readMessageCopy(stream, *buffer);

            mBuffer = std::shared_ptr<capnp::MallocMessageBuilder>(buffer);
            mPointer = mBuffer->template getRoot<T>().asReader();
        }

        /// Construct an owned version of a capnproto type `T` by deep-copying from a reader.
        explicit Shared(const typename T::Reader& reader): Shared() {
            auto buffer = new capnp::MallocMessageBuilder();
            buffer->setRoot(reader);

            mBuffer = std::shared_ptr<capnp::MallocMessageBuilder>(buffer);
            mPointer = mBuffer->template getRoot<T>().asReader();
        }

        /// Serialize the capnproto type `T` into the stream.  
        void serialize(kj::OutputStream& stream) const {
            capnp::writeMessage(stream, *this->mBuffer);
        }

        /// Return a T::Reader.
        typename T::Reader reader() const {
            return this->mPointer;
        }

        const typename T::Reader* operator->() const {
            return &mPointer;
        }

        bool operator==(std::nullptr_t) const {
            return this->mBuffer == nullptr;
        }

        bool operator!=(std::nullptr_t) const {
            return this->mBuffer != nullptr;
        }

        bool operator==(const Shared<T>& other) const {
            return this->mBuffer == other.mBuffer;
        }

        bool operator!=(const Shared<T>& other) const {
            return this->mBuffer != other.mBuffer;
        }

    protected:
        std::shared_ptr<capnp::MessageBuilder> mBuffer;
        typename T::Reader mPointer;
    };

}

#endif