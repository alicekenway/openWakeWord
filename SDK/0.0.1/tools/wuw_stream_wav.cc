#include <algorithm>
#include <iostream>
#include "tool_common.h"
int main(int argc,char**argv){try{if(argc!=3){std::cerr<<"usage: wuw_stream_wav MODEL_DIR AUDIO.wav\n";return 2;}auto pcm=tool::ReadWav(argv[2]);wuw_sdk::Engine engine(argv[1]);auto stream=engine.CreateStream();stream.BeginSegment(1,0);for(size_t p=0;p<pcm.size();p+=1600)stream.AcceptPcm16(pcm.data()+p,std::min<size_t>(1600,pcm.size()-p));stream.EndSegment();auto events=stream.ReadEvents();for(const auto&e:events)std::cout<<e.keyword_id<<"\t"<<e.confidence<<"\t"<<e.start_sample_index<<"\t"<<e.end_sample_index<<"\n";std::cerr<<"samples="<<pcm.size()<<" events="<<events.size()<<"\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
