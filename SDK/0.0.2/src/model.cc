#include "internal.h"

#include <algorithm>
#include <cstring>

namespace wuw {
namespace {
size_t Product(const std::vector<int64_t>&shape){size_t n=1;for(auto x:shape){if(x<0)throw Error(WUW_SDK_STATUS_INCOMPATIBLE_MODEL,"unexpected dynamic output dimension");n*=static_cast<size_t>(x);}return n;}
void RequireFile(const std::filesystem::path&p){if(!std::filesystem::is_regular_file(p))throw Error(WUW_SDK_STATUS_NOT_FOUND,"missing model file: "+p.string());}
}
Model::Model(const std::filesystem::path&dir,const Config&config,int intra,int inter):env_(ORT_LOGGING_LEVEL_WARNING,"wuw_sdk"),contract_(config.contract){
  auto p1=dir/"stage1.onnx",p2=dir/"stage2.onnx";RequireFile(p1);RequireFile(p2);
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  session_options_.SetIntraOpNumThreads(intra>0?intra:1);session_options_.SetInterOpNumThreads(inter>0?inter:1);
  try{stage1_=Ort::Session(env_,p1.c_str(),session_options_);stage2_=Ort::Session(env_,p2.c_str(),session_options_);}catch(const Ort::Exception&e){throw Error(WUW_SDK_STATUS_INCOMPATIBLE_MODEL,std::string("cannot load ONNX model: ")+e.what());}
}
Stage1Output Model::RunStage1(const std::vector<float>&features,int frames,int64_t offset,const std::vector<float>&att_cache,const std::vector<float>&cnn_cache,const std::vector<uint8_t>&att_mask,std::vector<float>*next_att,std::vector<float>*next_cnn)const{
  Ort::MemoryInfo memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);int64_t required=contract_.required_cache_size;
  std::array<int64_t,3> feature_shape={1,frames,contract_.feature_dim};std::array<int64_t,4> att_shape={contract_.num_layers,contract_.num_heads,contract_.required_cache_size,contract_.head_dim};std::array<int64_t,4> cnn_shape={contract_.num_layers,1,contract_.cnn_channels,contract_.cnn_cache};std::array<int64_t,3>mask_shape={1,1,contract_.required_cache_size+16};
  std::vector<Ort::Value>inputs;inputs.reserve(6);inputs.push_back(Ort::Value::CreateTensor<float>(memory,const_cast<float*>(features.data()),features.size(),feature_shape.data(),feature_shape.size()));inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory,&offset,1,nullptr,0));inputs.push_back(Ort::Value::CreateTensor<int64_t>(memory,&required,1,nullptr,0));inputs.push_back(Ort::Value::CreateTensor<float>(memory,const_cast<float*>(att_cache.data()),att_cache.size(),att_shape.data(),att_shape.size()));inputs.push_back(Ort::Value::CreateTensor<float>(memory,const_cast<float*>(cnn_cache.data()),cnn_cache.size(),cnn_shape.data(),cnn_shape.size()));inputs.push_back(Ort::Value::CreateTensor<bool>(memory,reinterpret_cast<bool*>(const_cast<uint8_t*>(att_mask.data())),att_mask.size(),mask_shape.data(),mask_shape.size()));
  const char*names[]={"chunk","offset","required_cache_size","att_cache","cnn_cache","att_mask"};const char*out_names[]={"encoder_out","ctc_log_probs","next_att_cache","next_cnn_cache"};
  try {
    auto out=stage1_.Run(Ort::RunOptions{nullptr},names,inputs.data(),inputs.size(),out_names,4);
    auto es=out[0].GetTensorTypeAndShapeInfo().GetShape();
    auto cs=out[1].GetTensorTypeAndShapeInfo().GetShape();
    if(es.size()!=3||cs.size()!=3||es[0]!=1||cs[0]!=1||es[1]!=cs[1]||es[2]!=contract_.encoder_dim||cs[2]!=contract_.vocabulary_size)
      throw Error(WUW_SDK_STATUS_INCOMPATIBLE_MODEL,"unexpected stage-1 output shapes");
    Stage1Output result;result.frames=static_cast<int>(es[1]);
    result.encoder.assign(out[0].GetTensorData<float>(),out[0].GetTensorData<float>()+Product(es));
    result.ctc.assign(out[1].GetTensorData<float>(),out[1].GetTensorData<float>()+Product(cs));
    auto as=out[2].GetTensorTypeAndShapeInfo().GetShape();
    auto ns=out[3].GetTensorTypeAndShapeInfo().GetShape();
    next_att->assign(out[2].GetTensorData<float>(),out[2].GetTensorData<float>()+Product(as));
    next_cnn->assign(out[3].GetTensorData<float>(),out[3].GetTensorData<float>()+Product(ns));
    if(next_att->size()!=att_cache.size()||next_cnn->size()!=cnn_cache.size())
      throw Error(WUW_SDK_STATUS_INCOMPATIBLE_MODEL,
        "stage-1 cache output shape changed at offset "+std::to_string(offset)+
        ": attention "+std::to_string(next_att->size())+" vs "+std::to_string(att_cache.size())+
        ", cnn "+std::to_string(next_cnn->size())+" vs "+std::to_string(cnn_cache.size()));
    return result;
  } catch(const Error&) { throw; }
    catch(const Ort::Exception&e) { throw Error(WUW_SDK_STATUS_INTERNAL_ERROR,std::string("stage-1 inference failed: ")+e.what()); }
}
float Model::RunStage2(const std::vector<float>&encoder,int frames,float top,float margin,int winner,int count)const{
  Ort::MemoryInfo memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);std::vector<float>mask(frames,1),onehot(count,0);onehot.at(static_cast<size_t>(winner))=1;std::array<int64_t,3>es={1,frames,contract_.encoder_dim};std::array<int64_t,2>ms={1,frames},scalar={1,1},ks={1,count};std::vector<Ort::Value>in;in.push_back(Ort::Value::CreateTensor<float>(memory,const_cast<float*>(encoder.data()),encoder.size(),es.data(),3));in.push_back(Ort::Value::CreateTensor<float>(memory,mask.data(),mask.size(),ms.data(),2));in.push_back(Ort::Value::CreateTensor<float>(memory,&top,1,scalar.data(),2));in.push_back(Ort::Value::CreateTensor<float>(memory,&margin,1,scalar.data(),2));in.push_back(Ort::Value::CreateTensor<float>(memory,onehot.data(),onehot.size(),ks.data(),2));const char*names[]={"encoder_features","frame_mask","top_score","margin","winner_onehot"};const char*out_names[]={"wake_probability"};try{auto out=stage2_.Run(Ort::RunOptions{nullptr},names,in.data(),in.size(),out_names,1);return out[0].GetTensorData<float>()[0];}catch(const Ort::Exception&e){throw Error(WUW_SDK_STATUS_INTERNAL_ERROR,std::string("stage-2 inference failed: ")+e.what());}
}
EngineImpl::EngineImpl(const std::filesystem::path&dir,const WuwSdkEngineOptions&options){if(options.verify_checksums)VerifyChecksums(dir);config=LoadConfig(dir,options);model=std::make_shared<Model>(dir,config,options.intra_op_num_threads,options.inter_op_num_threads);}
}  // namespace wuw
