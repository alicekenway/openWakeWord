#ifndef WUW_SDK_C_API_H_
#define WUW_SDK_C_API_H_

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define WUW_SDK_EXPORT __declspec(dllexport)
#else
#define WUW_SDK_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define WUW_SDK_ABI_VERSION 1u
#define WUW_SDK_KEYWORD_ID_CAPACITY 64u
#define WUW_SDK_DISPLAY_TEXT_CAPACITY 96u

typedef struct WuwSdkEngine WuwSdkEngine;
typedef struct WuwSdkStream WuwSdkStream;

typedef enum WuwSdkStatus {
  WUW_SDK_STATUS_OK = 0,
  WUW_SDK_STATUS_INVALID_ARGUMENT = 1,
  WUW_SDK_STATUS_INVALID_STATE = 2,
  WUW_SDK_STATUS_NOT_FOUND = 3,
  WUW_SDK_STATUS_INCOMPATIBLE_MODEL = 4,
  WUW_SDK_STATUS_IO_ERROR = 5,
  WUW_SDK_STATUS_RESOURCE_EXHAUSTED = 6,
  WUW_SDK_STATUS_INTERNAL_ERROR = 7
} WuwSdkStatus;

typedef struct WuwSdkKeywordThreshold {
  uint32_t struct_size;
  const char* keyword_id;
  float threshold;
} WuwSdkKeywordThreshold;

typedef struct WuwSdkEngineOptions {
  uint32_t struct_size;
  uint32_t abi_version;
  const char* threshold_config_path;
  const WuwSdkKeywordThreshold* stage1_overrides;
  size_t stage1_override_count;
  float stage2_threshold_override; /* NaN keeps the configured default. */
  int32_t intra_op_num_threads;
  int32_t inter_op_num_threads;
  uint8_t verify_checksums;
  uint8_t reserved[7];
} WuwSdkEngineOptions;

typedef struct WuwSdkKeywordInfo {
  uint32_t struct_size;
  char keyword_id[WUW_SDK_KEYWORD_ID_CAPACITY];
  char display_text[WUW_SDK_DISPLAY_TEXT_CAPACITY];
  float stage1_threshold;
} WuwSdkKeywordInfo;

typedef struct WuwSdkEvent {
  uint32_t struct_size;
  uint64_t segment_id;
  int64_t start_sample_index;
  int64_t end_sample_index;
  int64_t detection_sample_index;
  float confidence;
  char keyword_id[WUW_SDK_KEYWORD_ID_CAPACITY];
  char display_text[WUW_SDK_DISPLAY_TEXT_CAPACITY];
} WuwSdkEvent;

typedef struct WuwSdkStreamStats {
  uint32_t struct_size;
  uint64_t segments_started;
  uint64_t samples_accepted;
  uint64_t stage1_chunks;
  uint64_t stage1_candidates;
  uint64_t stage2_runs;
  uint64_t detections;
  uint64_t processing_time_us;
  uint64_t max_accept_time_us;
} WuwSdkStreamStats;

WUW_SDK_EXPORT void wuw_sdk_engine_options_init(WuwSdkEngineOptions* options);

WUW_SDK_EXPORT WuwSdkStatus wuw_sdk_engine_create(
    const char* model_dir, const WuwSdkEngineOptions* options,
    WuwSdkEngine** engine);
WUW_SDK_EXPORT void wuw_sdk_engine_destroy(WuwSdkEngine* engine);

WUW_SDK_EXPORT size_t wuw_sdk_engine_get_keyword_count(
    const WuwSdkEngine* engine);
WUW_SDK_EXPORT WuwSdkStatus wuw_sdk_engine_get_keyword_info(
    const WuwSdkEngine* engine, size_t index, WuwSdkKeywordInfo* info);
WUW_SDK_EXPORT float wuw_sdk_engine_get_stage2_threshold(
    const WuwSdkEngine* engine);

WUW_SDK_EXPORT WuwSdkStatus wuw_sdk_stream_create(
    WuwSdkEngine* engine, WuwSdkStream** stream);
WUW_SDK_EXPORT void wuw_sdk_stream_destroy(WuwSdkStream* stream);
WUW_SDK_EXPORT WuwSdkStatus wuw_sdk_stream_begin_segment(
    WuwSdkStream* stream, uint64_t segment_id, int64_t first_sample_index);
WUW_SDK_EXPORT WuwSdkStatus wuw_sdk_stream_accept_pcm16(
    WuwSdkStream* stream, const int16_t* samples, size_t sample_count);
WUW_SDK_EXPORT WuwSdkStatus wuw_sdk_stream_end_segment(WuwSdkStream* stream);
WUW_SDK_EXPORT WuwSdkStatus wuw_sdk_stream_read_events(
    WuwSdkStream* stream, WuwSdkEvent* events, size_t capacity,
    size_t* written);
WUW_SDK_EXPORT WuwSdkStatus wuw_sdk_stream_reset(WuwSdkStream* stream);
WUW_SDK_EXPORT WuwSdkStatus wuw_sdk_stream_get_stats(
    const WuwSdkStream* stream, WuwSdkStreamStats* stats);

WUW_SDK_EXPORT const char* wuw_sdk_last_error_message(const void* handle);
WUW_SDK_EXPORT const char* wuw_sdk_version(void);
WUW_SDK_EXPORT uint32_t wuw_sdk_abi_version(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* WUW_SDK_C_API_H_ */
