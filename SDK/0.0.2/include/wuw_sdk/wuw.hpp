#ifndef WUW_SDK_WUW_HPP_
#define WUW_SDK_WUW_HPP_

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "wuw_sdk/c_api.h"

namespace wuw_sdk {

class Error : public std::runtime_error {
 public:
  Error(WuwSdkStatus status, const std::string& message)
      : std::runtime_error(message), status_(status) {}
  WuwSdkStatus status() const noexcept { return status_; }

 private:
  WuwSdkStatus status_;
};

inline void ThrowIfError(WuwSdkStatus status, const void* handle) {
  if (status != WUW_SDK_STATUS_OK) {
    throw Error(status, wuw_sdk_last_error_message(handle));
  }
}

class Stream;

class Engine {
 public:
  explicit Engine(const std::string& model_dir,
                  const WuwSdkEngineOptions* options = nullptr) {
    WuwSdkEngine* value = nullptr;
    const auto status = wuw_sdk_engine_create(model_dir.c_str(), options, &value);
    if (status != WUW_SDK_STATUS_OK) {
      throw Error(status, wuw_sdk_last_error_message(value));
    }
    engine_.reset(value);
  }

  size_t keyword_count() const {
    return wuw_sdk_engine_get_keyword_count(engine_.get());
  }

  WuwSdkKeywordInfo keyword(size_t index) const {
    WuwSdkKeywordInfo info{};
    info.struct_size = sizeof(info);
    ThrowIfError(wuw_sdk_engine_get_keyword_info(engine_.get(), index, &info),
                 engine_.get());
    return info;
  }

  float stage2_threshold() const {
    return wuw_sdk_engine_get_stage2_threshold(engine_.get());
  }

  Stream CreateStream();

 private:
  struct Deleter {
    void operator()(WuwSdkEngine* value) const {
      wuw_sdk_engine_destroy(value);
    }
  };
  std::unique_ptr<WuwSdkEngine, Deleter> engine_;
  friend class Stream;
};

class Stream {
 public:
  explicit Stream(WuwSdkEngine* engine) {
    WuwSdkStream* value = nullptr;
    ThrowIfError(wuw_sdk_stream_create(engine, &value), engine);
    stream_.reset(value);
  }

  void BeginSegment(uint64_t id, int64_t first_sample_index) {
    CheckThread();
    ThrowIfError(wuw_sdk_stream_begin_segment(stream_.get(), id,
                                               first_sample_index),
                 stream_.get());
  }

  void AcceptPcm16(const int16_t* samples, size_t count) {
    CheckThread();
    ThrowIfError(wuw_sdk_stream_accept_pcm16(stream_.get(), samples, count),
                 stream_.get());
  }

  void EndSegment() {
    CheckThread();
    ThrowIfError(wuw_sdk_stream_end_segment(stream_.get()), stream_.get());
  }

  std::vector<WuwSdkEvent> ReadEvents(size_t capacity = 32) {
    CheckThread();
    std::vector<WuwSdkEvent> values(capacity);
    for (auto& item : values) item.struct_size = sizeof(item);
    size_t written = 0;
    ThrowIfError(wuw_sdk_stream_read_events(stream_.get(), values.data(),
                                             values.size(), &written),
                 stream_.get());
    values.resize(written);
    return values;
  }

  void Reset() {
    ThrowIfError(wuw_sdk_stream_reset(stream_.get()), stream_.get());
#ifndef NDEBUG
    // Reset is the explicit lifecycle boundary at which ownership may move to
    // a newly created worker thread. It must not race with another call.
    owner_thread_ = std::this_thread::get_id();
#endif
  }

  WuwSdkStreamStats Stats() const {
    CheckThread();
    WuwSdkStreamStats stats{};
    stats.struct_size = sizeof(stats);
    ThrowIfError(wuw_sdk_stream_get_stats(stream_.get(), &stats), stream_.get());
    return stats;
  }

 private:
  void CheckThread() const {
#ifndef NDEBUG
    const std::thread::id current = std::this_thread::get_id();
    if (owner_thread_ == std::thread::id()) owner_thread_ = current;
    if (owner_thread_ != current) {
      throw std::logic_error("one WUW Stream cannot be used by multiple threads");
    }
#endif
  }

  struct Deleter {
    void operator()(WuwSdkStream* value) const {
      wuw_sdk_stream_destroy(value);
    }
  };
  std::unique_ptr<WuwSdkStream, Deleter> stream_;
#ifndef NDEBUG
  mutable std::thread::id owner_thread_;
#endif
};

inline Stream Engine::CreateStream() { return Stream(engine_.get()); }

}  // namespace wuw_sdk

#endif  // WUW_SDK_WUW_HPP_
