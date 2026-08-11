#include <algorithm>
#include <fstream>
#include <iostream>
#include "tool_common.h"

int main(int argc,char**argv){
  try{
    if(argc<4){std::cerr<<"usage: wuw_eval_manifest MODEL_DIR INPUT.jsonl OUTPUT.jsonl [TASK_INDEX TASK_COUNT]\n";return 2;}
    int task=argc>4?std::stoi(argv[4]):0,count=argc>5?std::stoi(argv[5]):1;
    if(task<0||count<1||task>=count)throw std::runtime_error("invalid shard");
    std::ifstream in(argv[2]);std::ofstream out(argv[3]);if(!in||!out)throw std::runtime_error("cannot open manifest/output");
    wuw_sdk::Engine engine(argv[1]);auto stream=engine.CreateStream();std::string line;uint64_t row=0,done=0;
    while(std::getline(in,line)){
      uint64_t index=row++;if(index%count!=static_cast<uint64_t>(task))continue;
      std::string path=tool::JsonString(line,"path");if(path.empty())path=tool::JsonString(line,"audio_path");if(path.empty())path=tool::JsonString(line,"audio_file");
      std::string id=tool::JsonString(line,"id"),expected=tool::JsonString(line,"expected_keyword_id"),transcript=tool::JsonString(line,"text");
      try{
        auto pcm=tool::ReadWav(path);stream.Reset();stream.BeginSegment(index,0);
        for(size_t p=0;p<pcm.size();p+=1600)stream.AcceptPcm16(pcm.data()+p,std::min<size_t>(1600,pcm.size()-p));
        stream.EndSegment();auto events=stream.ReadEvents();auto stats=stream.Stats();
        double rtf=stats.processing_time_us/(1000000.0*pcm.size()/16000.0);
        out<<"{\"source_index\":"<<index<<",\"id\":\""<<tool::Escape(id)<<"\",\"path\":\""<<tool::Escape(path)<<"\",\"text\":\""<<tool::Escape(transcript)<<"\",\"expected_keyword_id\":\""<<tool::Escape(expected)<<"\",\"samples\":"<<pcm.size()<<",\"processing_time_us\":"<<stats.processing_time_us<<",\"rtf\":"<<rtf<<",\"events\":[";
        for(size_t i=0;i<events.size();++i){if(i)out<<',';out<<"{\"keyword_id\":\""<<tool::Escape(events[i].keyword_id)<<"\",\"confidence\":"<<events[i].confidence<<",\"start_sample\":"<<events[i].start_sample_index<<",\"end_sample\":"<<events[i].end_sample_index<<'}';}
        out<<"]}\n";
      }catch(const std::exception&e){out<<"{\"source_index\":"<<index<<",\"path\":\""<<tool::Escape(path)<<"\",\"error\":\""<<tool::Escape(e.what())<<"\"}\n";}
      ++done;
    }
    std::cerr<<"task="<<task<<" rows="<<done<<"\n";return 0;
  }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
