#include <algorithm>
#include <fstream>
#include <iostream>
#include "tool_common.h"

int main(int argc,char**argv){
  try{
    if(argc<4){std::cerr<<"usage: wuw_eval_manifest MODEL_DIR INPUT.jsonl OUTPUT.jsonl [TASK_INDEX TASK_COUNT [WINDOW_SAMPLES STRIDE_SAMPLES]]\n";return 2;}
    int task=argc>4?std::stoi(argv[4]):0,count=argc>5?std::stoi(argv[5]):1;
    size_t window_samples=argc>6?std::stoull(argv[6]):0,stride_samples=argc>7?std::stoull(argv[7]):window_samples;
    if(task<0||count<1||task>=count)throw std::runtime_error("invalid shard");
    if((window_samples&&!stride_samples)||stride_samples>window_samples)throw std::runtime_error("invalid window/stride");
    std::ifstream in(argv[2]);std::ofstream out(argv[3]);if(!in||!out)throw std::runtime_error("cannot open manifest/output");
    wuw_sdk::Engine engine(argv[1]);auto stream=engine.CreateStream();std::string line;uint64_t row=0,done=0;
    while(std::getline(in,line)){
      uint64_t index=row++;if(index%count!=static_cast<uint64_t>(task))continue;
      std::string path=tool::JsonString(line,"path");if(path.empty())path=tool::JsonString(line,"audio_path");if(path.empty())path=tool::JsonString(line,"audio_file");
      std::string id=tool::JsonString(line,"id"),expected=tool::JsonString(line,"expected_keyword_id"),transcript=tool::JsonString(line,"text");
      try{
        tool::WavReader wav(path);std::vector<WuwSdkEvent>events;uint64_t processing_us=0,window_id=0;
        auto process=[&](const int16_t*data,size_t samples,int64_t first){stream.Reset();stream.BeginSegment((index<<20)+window_id++,first);for(size_t p=0;p<samples;p+=1600)stream.AcceptPcm16(data+p,std::min<size_t>(1600,samples-p));stream.EndSegment();auto found=stream.ReadEvents(64);events.insert(events.end(),found.begin(),found.end());processing_us+=stream.Stats().processing_time_us;};
        if(!window_samples){std::vector<int16_t>pcm(1600);size_t n=0;stream.Reset();stream.BeginSegment(index,0);while((n=wav.Read(pcm.data(),pcm.size()))!=0)stream.AcceptPcm16(pcm.data(),n);stream.EndSegment();events=stream.ReadEvents(64);processing_us=stream.Stats().processing_time_us;}
        else{std::vector<int16_t>buffer;buffer.reserve(window_samples);std::vector<int16_t>chunk(1600);size_t n=0;int64_t first=0;while((n=wav.Read(chunk.data(),chunk.size()))!=0){buffer.insert(buffer.end(),chunk.begin(),chunk.begin()+static_cast<std::ptrdiff_t>(n));while(buffer.size()>=window_samples){process(buffer.data(),window_samples,first);buffer.erase(buffer.begin(),buffer.begin()+static_cast<std::ptrdiff_t>(stride_samples));first+=stride_samples;}}if(!buffer.empty())process(buffer.data(),buffer.size(),first);}
        double rtf=processing_us/(1000000.0*wav.total_samples()/16000.0);
        out<<"{\"source_index\":"<<index<<",\"id\":\""<<tool::Escape(id)<<"\",\"path\":\""<<tool::Escape(path)<<"\",\"text\":\""<<tool::Escape(transcript)<<"\",\"expected_keyword_id\":\""<<tool::Escape(expected)<<"\",\"samples\":"<<wav.total_samples()<<",\"processing_time_us\":"<<processing_us<<",\"rtf\":"<<rtf<<",\"events\":[";
        for(size_t i=0;i<events.size();++i){if(i)out<<',';out<<"{\"keyword_id\":\""<<tool::Escape(events[i].keyword_id)<<"\",\"confidence\":"<<events[i].confidence<<",\"start_sample\":"<<events[i].start_sample_index<<",\"end_sample\":"<<events[i].end_sample_index<<'}';}
        out<<"]}\n";
      }catch(const std::exception&e){out<<"{\"source_index\":"<<index<<",\"path\":\""<<tool::Escape(path)<<"\",\"error\":\""<<tool::Escape(e.what())<<"\"}\n";}
      ++done;
    }
    std::cerr<<"task="<<task<<" rows="<<done<<"\n";return 0;
  }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
