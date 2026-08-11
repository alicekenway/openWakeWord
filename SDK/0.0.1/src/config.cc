#include "internal.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>

namespace wuw {
namespace {

class Parser {
 public:
  explicit Parser(const std::string& text) : text_(text) {}
  Json Parse() {
    Json result = Value();
    Space();
    if (pos_ != text_.size()) Fail("trailing data");
    return result;
  }

 private:
  [[noreturn]] void Fail(const std::string& why) const {
    throw Error(WUW_SDK_STATUS_INCOMPATIBLE_MODEL,
                "invalid JSON at byte " + std::to_string(pos_) + ": " + why);
  }
  void Space() { while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_; }
  bool Take(char c) { Space(); if (pos_ < text_.size() && text_[pos_] == c) { ++pos_; return true; } return false; }
  Json Value() {
    Space();
    if (pos_ >= text_.size()) Fail("expected value");
    if (text_[pos_] == '{') return Object();
    if (text_[pos_] == '[') return Array();
    if (text_[pos_] == '"') return Json{String()};
    if (text_.compare(pos_, 4, "true") == 0) { pos_ += 4; return Json{true}; }
    if (text_.compare(pos_, 5, "false") == 0) { pos_ += 5; return Json{false}; }
    if (text_.compare(pos_, 4, "null") == 0) { pos_ += 4; return Json{nullptr}; }
    return Json{Number()};
  }
  Json Object() {
    Take('{'); Json::Object out;
    if (Take('}')) return Json{out};
    for (;;) {
      Space(); if (pos_ >= text_.size() || text_[pos_] != '"') Fail("expected object key");
      std::string key = String();
      if (!Take(':')) Fail("expected ':'");
      if (!out.emplace(key, Value()).second) Fail("duplicate object key");
      if (Take('}')) break;
      if (!Take(',')) Fail("expected ','");
    }
    return Json{out};
  }
  Json Array() {
    Take('['); Json::Array out;
    if (Take(']')) return Json{out};
    for (;;) {
      out.push_back(Value());
      if (Take(']')) break;
      if (!Take(',')) Fail("expected ','");
    }
    return Json{out};
  }
  std::string String() {
    if (text_[pos_++] != '"') Fail("expected string");
    std::string out;
    while (pos_ < text_.size()) {
      char c = text_[pos_++];
      if (c == '"') return out;
      if (static_cast<unsigned char>(c) < 0x20) Fail("control character in string");
      if (c != '\\') { out.push_back(c); continue; }
      if (pos_ >= text_.size()) Fail("unfinished escape");
      c = text_[pos_++];
      switch (c) {
        case '"': case '\\': case '/': out.push_back(c); break;
        case 'b': out.push_back('\b'); break; case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break; case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        default: Fail("unsupported escape (bundle metadata must use UTF-8 directly)");
      }
    }
    Fail("unterminated string");
  }
  double Number() {
    Space(); size_t start = pos_;
    if (pos_ < text_.size() && text_[pos_] == '-') ++pos_;
    while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    if (pos_ < text_.size() && text_[pos_] == '.') { ++pos_; while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_; }
    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_; if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) ++pos_;
      while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    }
    if (start == pos_) Fail("expected number");
    try { size_t used = 0; double v = std::stod(text_.substr(start, pos_ - start), &used); if (used != pos_ - start || !std::isfinite(v)) Fail("invalid number"); return v; }
    catch (const std::exception&) { Fail("invalid number"); }
  }
  const std::string& text_; size_t pos_ = 0;
};

int Int(const Json& j, const char* name) {
  double v = j.number();
  if (std::floor(v) != v || v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max())
    throw Error(WUW_SDK_STATUS_INCOMPATIBLE_MODEL, std::string(name) + " must be an integer");
  return static_cast<int>(v);
}
float Float(const Json& j, const char* name) {
  double v = j.number();
  if (!std::isfinite(v)) throw Error(WUW_SDK_STATUS_INCOMPATIBLE_MODEL, std::string(name) + " must be finite");
  return static_cast<float>(v);
}
const Json* MaybeNested(const Json& root, const char* outer, const char* inner) {
  const Json* o = root.find(outer); return o ? o->find(inner) : nullptr;
}
void ApplyThresholdFile(const std::filesystem::path& path, Config* config) {
  Json root = ReadJson(path);
  if (const Json* v = root.find("stage2_threshold")) config->stage2_threshold = Float(*v, "stage2_threshold");
  const Json* entries = root.find("keywords");
  if (!entries) return;
  std::unordered_map<std::string, float> values;
  for (const Json& item : entries->array()) values[item.at("id").string()] = Float(item.at("threshold"), "threshold");
  for (Keyword& keyword : config->keywords) {
    auto it = values.find(keyword.id); if (it != values.end()) keyword.threshold = it->second;
  }
}

}  // namespace

const Json::Object& Json::object() const { if (auto p = std::get_if<Object>(&value)) return *p; throw Error(WUW_SDK_STATUS_INCOMPATIBLE_MODEL, "expected JSON object"); }
const Json::Array& Json::array() const { if (auto p = std::get_if<Array>(&value)) return *p; throw Error(WUW_SDK_STATUS_INCOMPATIBLE_MODEL, "expected JSON array"); }
const std::string& Json::string() const { if (auto p = std::get_if<std::string>(&value)) return *p; throw Error(WUW_SDK_STATUS_INCOMPATIBLE_MODEL, "expected JSON string"); }
double Json::number() const { if (auto p = std::get_if<double>(&value)) return *p; throw Error(WUW_SDK_STATUS_INCOMPATIBLE_MODEL, "expected JSON number"); }
bool Json::boolean() const { if (auto p = std::get_if<bool>(&value)) return *p; throw Error(WUW_SDK_STATUS_INCOMPATIBLE_MODEL, "expected JSON boolean"); }
const Json& Json::at(const std::string& key) const { auto it = object().find(key); if (it == object().end()) throw Error(WUW_SDK_STATUS_INCOMPATIBLE_MODEL, "missing JSON field: " + key); return it->second; }
const Json* Json::find(const std::string& key) const { auto p = std::get_if<Object>(&value); if (!p) return nullptr; auto it = p->find(key); return it == p->end() ? nullptr : &it->second; }
Json ParseJson(const std::string& text) { return Parser(text).Parse(); }
std::string ReadFile(const std::filesystem::path& path) { std::ifstream in(path, std::ios::binary); if (!in) throw Error(WUW_SDK_STATUS_NOT_FOUND, "cannot open " + path.string()); std::ostringstream ss; ss << in.rdbuf(); if (!in.good() && !in.eof()) throw Error(WUW_SDK_STATUS_IO_ERROR, "cannot read " + path.string()); return ss.str(); }
Json ReadJson(const std::filesystem::path& path) { try { return ParseJson(ReadFile(path)); } catch (const Error& e) { throw Error(e.status, path.string() + ": " + e.what()); } }

Config LoadConfig(const std::filesystem::path& dir, const WuwSdkEngineOptions& options) {
  Config c;
  Json manifest = ReadJson(dir / "manifest.json");
  if (Int(manifest.at("bundle_schema_version"), "bundle_schema_version") != 1)
    throw Error(WUW_SDK_STATUS_INCOMPATIBLE_MODEL, "unsupported bundle_schema_version");
  Json contract = ReadJson(dir / "stage1_contract.json");
  c.contract.sample_rate = Int(contract.at("sample_rate"), "sample_rate");
  c.contract.feature_dim = Int(contract.at("fbank").at("num_mel_bins"), "num_mel_bins");
  c.contract.frame_length = static_cast<int>(std::lround(contract.at("fbank").at("frame_length_ms").number() * c.contract.sample_rate / 1000.0));
  c.contract.frame_shift = static_cast<int>(std::lround(contract.at("fbank").at("frame_shift_ms").number() * c.contract.sample_rate / 1000.0));
  c.contract.chunk_frames = Int(contract.at("chunk_frames"), "chunk_frames");
  if (const Json* v = contract.find("chunk_stride_frames")) c.contract.chunk_stride = Int(*v, "chunk_stride_frames");
  if (const Json* v = contract.find("minimum_input_frames")) c.contract.minimum_chunk_frames = Int(*v, "minimum_input_frames");
  if (const Json* v = contract.find("encoder_output_size")) c.contract.encoder_dim = Int(*v, "encoder_output_size");
  if (const Json* v = contract.find("vocab_size")) c.contract.vocabulary_size = Int(*v, "vocab_size");
  if (const Json* v = contract.find("blank_id")) c.contract.blank_id = Int(*v, "blank_id");
  if (const Json* v = contract.find("initial_offset")) c.contract.initial_offset = Int(*v, "initial_offset");
  if (const Json* v = contract.find("encoder_frame_shift_ms")) c.contract.encoder_frame_samples = static_cast<int>(std::lround(v->number() * c.contract.sample_rate / 1000.0));
  if (const Json* v = MaybeNested(contract, "constant_inputs", "required_cache_size")) c.contract.required_cache_size = Int(*v, "required_cache_size");
  if (const Json* caches = contract.find("cache_inputs")) {
    for (const Json& cache : caches->array()) {
      const auto& shape = cache.at("shape").array();
      if (cache.at("input").string() == "att_cache" && shape.size() == 4) {
        c.contract.num_layers=Int(shape[0],"att_cache shape"); c.contract.num_heads=Int(shape[1],"att_cache shape"); c.contract.head_dim=Int(shape[3],"att_cache shape");
      } else if (cache.at("input").string() == "cnn_cache" && shape.size() == 4) {
        c.contract.cnn_channels=Int(shape[2],"cnn_cache shape"); c.contract.cnn_cache=Int(shape[3],"cnn_cache shape");
      }
    }
  }
  if (c.contract.sample_rate != 16000 || c.contract.feature_dim <= 0 || c.contract.encoder_dim <= 0 || c.contract.vocabulary_size <= 1)
    throw Error(WUW_SDK_STATUS_INCOMPATIBLE_MODEL, "unsupported or invalid stage-1 contract");

  Json keys = ReadJson(dir / "keywords.json");
  std::set<std::string> ids;
  for (const Json& item : keys.at("keywords").array()) {
    Keyword k; k.id=item.at("id").string(); k.display=item.at("display_text").string(); k.threshold=Float(item.at("threshold"),"threshold");
    for (const Json& token : item.at("token_ids").array()) k.tokens.push_back(Int(token,"token_id"));
    if (k.id.empty() || k.id.size() >= WUW_SDK_KEYWORD_ID_CAPACITY || k.display.size() >= WUW_SDK_DISPLAY_TEXT_CAPACITY || k.tokens.empty() || !ids.insert(k.id).second)
      throw Error(WUW_SDK_STATUS_INCOMPATIBLE_MODEL, "invalid or duplicate keyword entry: " + k.id);
    for (int64_t token : k.tokens) if (token < 0 || token >= c.contract.vocabulary_size) throw Error(WUW_SDK_STATUS_INCOMPATIBLE_MODEL, "keyword token outside vocabulary");
    c.keywords.push_back(std::move(k));
  }
  Json defaults = ReadJson(dir / "sdk_defaults.json");
  if (const Json* v=defaults.find("stage2_threshold")) c.stage2_threshold=Float(*v,"stage2_threshold");
  if (const Json* v=defaults.find("proposal_floor")) c.proposal_floor=Float(*v,"proposal_floor");
  if (const Json* v=defaults.find("competitor_beam")) c.competitor_beam=Int(*v,"competitor_beam");
  if (const Json* v=defaults.find("token_prune")) c.token_prune=Int(*v,"token_prune");
  if (const Json* v=defaults.find("pre_margin_frames")) c.pre_margin=Int(*v,"pre_margin_frames");
  if (const Json* v=defaults.find("post_margin_frames")) c.post_margin=Int(*v,"post_margin_frames");
  if (const Json* v=defaults.find("max_search_frames")) c.max_search_frames=Int(*v,"max_search_frames");
  if (const Json* v=defaults.find("debounce_ms")) c.debounce_samples=static_cast<int64_t>(v->number()*c.contract.sample_rate/1000.0);
  if (const Json* v=defaults.find("max_segment_ms")) c.max_segment_samples=static_cast<int64_t>(v->number()*c.contract.sample_rate/1000.0);
  if (options.threshold_config_path && options.threshold_config_path[0]) ApplyThresholdFile(options.threshold_config_path, &c);
  for (size_t i=0; i<options.stage1_override_count; ++i) {
    const auto& o=options.stage1_overrides[i]; if (!o.keyword_id || o.struct_size < sizeof(WuwSdkKeywordThreshold)) throw Error(WUW_SDK_STATUS_INVALID_ARGUMENT,"invalid stage1 override");
    auto it=std::find_if(c.keywords.begin(),c.keywords.end(),[&](const Keyword& k){return k.id==o.keyword_id;});
    if (it==c.keywords.end()) throw Error(WUW_SDK_STATUS_INVALID_ARGUMENT,"unknown keyword override: "+std::string(o.keyword_id));
    it->threshold=o.threshold;
  }
  if (!std::isnan(options.stage2_threshold_override)) c.stage2_threshold=options.stage2_threshold_override;
  auto probability=[](float x){return std::isfinite(x)&&x>=0&&x<=1;};
  if (!probability(c.stage2_threshold)) throw Error(WUW_SDK_STATUS_INVALID_ARGUMENT,"stage2 threshold must be in [0,1]");
  for (const auto& k:c.keywords) if (!probability(k.threshold)) throw Error(WUW_SDK_STATUS_INVALID_ARGUMENT,"stage1 threshold must be in [0,1]");
  if (c.keywords.empty() || c.competitor_beam<1 || c.token_prune<1 || c.max_search_frames<1 || c.max_segment_samples<400) throw Error(WUW_SDK_STATUS_INCOMPATIBLE_MODEL,"invalid SDK defaults");
  return c;
}

}  // namespace wuw
