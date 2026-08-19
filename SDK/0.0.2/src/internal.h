#ifndef WUW_SDK_INTERNAL_H_
#define WUW_SDK_INTERNAL_H_

#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "wuw_sdk/c_api.h"

namespace wuw {

class Error : public std::runtime_error {
 public:
  Error(WuwSdkStatus status, const std::string& message)
      : std::runtime_error(message), status(status) {}
  WuwSdkStatus status;
};

struct Json {
  using Array = std::vector<Json>;
  using Object = std::map<std::string, Json>;
  std::variant<std::nullptr_t, bool, double, std::string, Array, Object> value;

  const Object& object() const;
  const Array& array() const;
  const std::string& string() const;
  double number() const;
  bool boolean() const;
  const Json& at(const std::string& key) const;
  const Json* find(const std::string& key) const;
};

Json ParseJson(const std::string& text);
Json ReadJson(const std::filesystem::path& path);
std::string ReadFile(const std::filesystem::path& path);
std::string Sha256File(const std::filesystem::path& path);
void VerifyChecksums(const std::filesystem::path& dir);
void CopyString(const std::string& source, char* destination, size_t capacity);

struct Keyword {
  std::string id;
  std::string display;
  std::vector<int64_t> tokens;
  float threshold = 0.5f;
};

struct Contract {
  int sample_rate = 16000;
  int feature_dim = 80;
  int frame_length = 400;
  int frame_shift = 160;
  int chunk_frames = 67;
  int chunk_stride = 64;
  int minimum_chunk_frames = 7;
  int encoder_dim = 192;
  int vocabulary_size = 78;
  int blank_id = 0;
  int encoder_frame_samples = 640;
  int initial_offset = 64;
  int required_cache_size = 64;
  int num_layers = 12;
  int num_heads = 4;
  int head_dim = 96;
  int cnn_channels = 192;
  int cnn_cache = 14;
};

struct Config {
  Contract contract;
  std::vector<Keyword> keywords;
  float stage2_threshold = 0.7f;
  float proposal_floor = -3.0f;
  int competitor_beam = 16;
  int token_prune = 8;
  int pre_margin = 3;
  int post_margin = 0;
  int max_search_frames = 128;
  int64_t debounce_samples = 16000;
  int64_t max_segment_samples = 16000 * 180;
};

Config LoadConfig(const std::filesystem::path& model_dir,
                  const WuwSdkEngineOptions& options);

class Fbank {
 public:
  Fbank(int sample_rate, int feature_dim, int frame_length, int frame_shift);
  void Reset();
  void Accept(const int16_t* samples, size_t count);
  std::vector<std::vector<float>> ReadFrames();

 private:
  std::vector<float> Compute(const int16_t* frame) const;
  int sample_rate_;
  int feature_dim_;
  int frame_length_;
  int frame_shift_;
  std::vector<int16_t> samples_;
  size_t next_frame_start_ = 0;
  std::vector<std::vector<float>> ready_;
  std::vector<std::vector<std::pair<int, float>>> mel_bins_;
  std::vector<float> window_;
};

struct Stage1Output {
  int frames = 0;
  std::vector<float> encoder;
  std::vector<float> ctc;
};

class Model {
 public:
  Model(const std::filesystem::path& model_dir, const Config& config,
        int intra_threads, int inter_threads);
  Stage1Output RunStage1(const std::vector<float>& features, int frames,
                         int64_t offset, const std::vector<float>& att_cache,
                         const std::vector<float>& cnn_cache,
                         const std::vector<uint8_t>& att_mask,
                         std::vector<float>* next_att_cache,
                         std::vector<float>* next_cnn_cache) const;
  float RunStage2(const std::vector<float>& encoder, int frames,
                  float top_score, float margin, int winner,
                  int keyword_count) const;

 private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  mutable Ort::Session stage1_{nullptr};
  mutable Ort::Session stage2_{nullptr};
  Contract contract_;
};

struct Trace {
  float score = -std::numeric_limits<float>::infinity();
  int64_t start = 0;
  int64_t end = -1;
};

class BoundedCtcScorer {
 public:
  BoundedCtcScorer(std::vector<int64_t> tokens, int blank_id, int horizon);
  void Reset();
  Trace Step(const float* log_probs, int vocabulary_size, int64_t frame);

 private:
  std::vector<int64_t> states_;
  size_t token_count_ = 0;
  int blank_id_;
  int horizon_;
  int64_t frame_ = 0;
  std::vector<float> scores_;
  std::vector<int64_t> ends_;
  std::vector<int64_t> starts_;
};

float KeywordVsFillerConfidence(const std::vector<float>& log_probs,
                                int frames, int vocab,
                                const std::vector<int64_t>& keyword,
                                int blank_id, int beam_size,
                                int token_prune);

struct InternalEvent {
  uint64_t segment_id = 0;
  int64_t start_sample = 0;
  int64_t end_sample = 0;
  int64_t detection_sample = 0;
  float confidence = 0;
  size_t keyword = 0;
};

class EngineImpl {
 public:
  EngineImpl(const std::filesystem::path& model_dir,
             const WuwSdkEngineOptions& options);
  Config config;
  std::shared_ptr<Model> model;
  std::string last_error;
};

class StreamImpl {
 public:
  explicit StreamImpl(std::shared_ptr<EngineImpl> engine);
  void Begin(uint64_t segment_id, int64_t first_sample);
  void Accept(const int16_t* samples, size_t count);
  void End();
  void Reset();
  size_t Read(WuwSdkEvent* events, size_t capacity);

  std::shared_ptr<EngineImpl> engine;
  WuwSdkStreamStats stats{};
  std::string last_error;

 private:
  void ProcessAvailable(bool final);
  void RunChunk(int frames);
  void ProcessEncoderFrame(const float* encoder, const float* ctc,
                           bool synthetic_eof);
  void MaybeCandidate(size_t winner, const Trace& trace, float top,
                      float margin, bool synthetic_eof);
  void ClearSegmentState();

  bool active_ = false;
  uint64_t segment_id_ = 0;
  int64_t segment_first_sample_ = 0;
  uint64_t segment_samples_ = 0;
  Fbank fbank_;
  std::vector<float> pending_features_;
  int pending_frames_ = 0;
  int64_t offset_ = 64;
  int chunks_run_ = 0;
  std::vector<float> att_cache_;
  std::vector<float> cnn_cache_;
  std::vector<std::unique_ptr<BoundedCtcScorer>> scorers_;
  std::vector<bool> previous_above_;
  std::vector<Trace> last_traces_;
  int64_t encoder_frame_ = 0;
  std::deque<std::pair<int64_t, std::vector<float>>> encoder_ring_;
  std::deque<std::pair<int64_t, std::vector<float>>> ctc_ring_;
  std::optional<std::pair<size_t, std::pair<int64_t, int64_t>>> last_candidate_;
  int64_t last_detection_end_ = std::numeric_limits<int64_t>::min() / 2;
  std::deque<InternalEvent> events_;
};

}  // namespace wuw

struct WuwSdkEngine { uint32_t magic = 0x57555745u; std::shared_ptr<wuw::EngineImpl> impl; };
struct WuwSdkStream { uint32_t magic = 0x57555753u; std::unique_ptr<wuw::StreamImpl> impl; };

#endif
