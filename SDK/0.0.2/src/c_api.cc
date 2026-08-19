#include "internal.h"

#include <cmath>
#include <cstring>

namespace { thread_local std::string g_error;
template<class F> WuwSdkStatus Guard(std::string*error,F&&f){try{f();if(error)error->clear();return WUW_SDK_STATUS_OK;}catch(const wuw::Error&e){if(error)*error=e.what();g_error=e.what();return e.status;}catch(const std::exception&e){if(error)*error=e.what();g_error=e.what();return WUW_SDK_STATUS_INTERNAL_ERROR;}catch(...){if(error)*error="unknown internal error";g_error="unknown internal error";return WUW_SDK_STATUS_INTERNAL_ERROR;}}
bool Engine(const WuwSdkEngine*e){return e&&e->magic==0x57555745u&&e->impl;}
bool Stream(const WuwSdkStream*s){return s&&s->magic==0x57555753u&&s->impl;}
}
extern "C" {
void wuw_sdk_engine_options_init(WuwSdkEngineOptions*o){if(!o)return;std::memset(o,0,sizeof(*o));o->struct_size=sizeof(*o);o->abi_version=WUW_SDK_ABI_VERSION;o->stage2_threshold_override=NAN;o->intra_op_num_threads=1;o->inter_op_num_threads=1;o->verify_checksums=1;}
WuwSdkStatus wuw_sdk_engine_create(const char*dir,const WuwSdkEngineOptions*provided,WuwSdkEngine**out){if(!out)return WUW_SDK_STATUS_INVALID_ARGUMENT;*out=nullptr;WuwSdkEngineOptions options;wuw_sdk_engine_options_init(&options);if(provided){if(provided->struct_size<sizeof(WuwSdkEngineOptions)||provided->abi_version!=WUW_SDK_ABI_VERSION){g_error="invalid engine options size or ABI version";return WUW_SDK_STATUS_INVALID_ARGUMENT;}options=*provided;}if(!dir||!*dir){g_error="model_dir is empty";return WUW_SDK_STATUS_INVALID_ARGUMENT;}return Guard(nullptr,[&]{auto handle=std::make_unique<WuwSdkEngine>();handle->impl=std::make_shared<wuw::EngineImpl>(dir,options);*out=handle.release();});}
void wuw_sdk_engine_destroy(WuwSdkEngine*e){if(e){e->magic=0;delete e;}}
size_t wuw_sdk_engine_get_keyword_count(const WuwSdkEngine*e){return Engine(e)?e->impl->config.keywords.size():0;}
WuwSdkStatus wuw_sdk_engine_get_keyword_info(const WuwSdkEngine*e,size_t i,WuwSdkKeywordInfo*out){if(!Engine(e)||!out||out->struct_size<sizeof(*out))return WUW_SDK_STATUS_INVALID_ARGUMENT;return Guard(&e->impl->last_error,[&]{if(i>=e->impl->config.keywords.size())throw wuw::Error(WUW_SDK_STATUS_INVALID_ARGUMENT,"keyword index is out of range");auto size=out->struct_size;std::memset(out,0,sizeof(*out));out->struct_size=size;const auto&k=e->impl->config.keywords[i];wuw::CopyString(k.id,out->keyword_id,sizeof(out->keyword_id));wuw::CopyString(k.display,out->display_text,sizeof(out->display_text));out->stage1_threshold=k.threshold;});}
float wuw_sdk_engine_get_stage2_threshold(const WuwSdkEngine*e){return Engine(e)?e->impl->config.stage2_threshold:NAN;}
WuwSdkStatus wuw_sdk_stream_create(WuwSdkEngine*e,WuwSdkStream**out){if(!Engine(e)||!out)return WUW_SDK_STATUS_INVALID_ARGUMENT;*out=nullptr;return Guard(&e->impl->last_error,[&]{auto h=std::make_unique<WuwSdkStream>();h->impl=std::make_unique<wuw::StreamImpl>(e->impl);*out=h.release();});}
void wuw_sdk_stream_destroy(WuwSdkStream*s){if(s){s->magic=0;delete s;}}
WuwSdkStatus wuw_sdk_stream_begin_segment(WuwSdkStream*s,uint64_t id,int64_t first){if(!Stream(s))return WUW_SDK_STATUS_INVALID_ARGUMENT;return Guard(&s->impl->last_error,[&]{s->impl->Begin(id,first);});}
WuwSdkStatus wuw_sdk_stream_accept_pcm16(WuwSdkStream*s,const int16_t*p,size_t n){if(!Stream(s))return WUW_SDK_STATUS_INVALID_ARGUMENT;return Guard(&s->impl->last_error,[&]{s->impl->Accept(p,n);});}
WuwSdkStatus wuw_sdk_stream_end_segment(WuwSdkStream*s){if(!Stream(s))return WUW_SDK_STATUS_INVALID_ARGUMENT;return Guard(&s->impl->last_error,[&]{s->impl->End();});}
WuwSdkStatus wuw_sdk_stream_read_events(WuwSdkStream*s,WuwSdkEvent*e,size_t cap,size_t*written){if(!Stream(s)||!written)return WUW_SDK_STATUS_INVALID_ARGUMENT;*written=0;return Guard(&s->impl->last_error,[&]{*written=s->impl->Read(e,cap);});}
WuwSdkStatus wuw_sdk_stream_reset(WuwSdkStream*s){if(!Stream(s))return WUW_SDK_STATUS_INVALID_ARGUMENT;return Guard(&s->impl->last_error,[&]{s->impl->Reset();});}
WuwSdkStatus wuw_sdk_stream_get_stats(const WuwSdkStream*s,WuwSdkStreamStats*out){if(!Stream(s)||!out||out->struct_size<sizeof(*out))return WUW_SDK_STATUS_INVALID_ARGUMENT;*out=s->impl->stats;return WUW_SDK_STATUS_OK;}
const char* wuw_sdk_last_error_message(const void*h){if(!h)return g_error.c_str();uint32_t magic=*static_cast<const uint32_t*>(h);if(magic==0x57555745u){auto e=static_cast<const WuwSdkEngine*>(h);return e->impl?e->impl->last_error.c_str():g_error.c_str();}if(magic==0x57555753u){auto s=static_cast<const WuwSdkStream*>(h);return s->impl?s->impl->last_error.c_str():g_error.c_str();}return g_error.c_str();}
const char* wuw_sdk_version(void){return "0.0.2";}uint32_t wuw_sdk_abi_version(void){return WUW_SDK_ABI_VERSION;}
}
