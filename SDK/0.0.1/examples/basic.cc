#include <cstdint>
#include <iostream>
#include <vector>
#include "wuw_sdk/wuw.hpp"

int main(int argc,char**argv){
  if(argc!=2){std::cerr<<"usage: basic MODEL_DIR\n";return 2;}
  try{
    wuw_sdk::Engine engine(argv[1]);
    auto stream=engine.CreateStream();
    std::vector<int16_t>pcm;  // Fill from the application's VAD/audio bank.
    stream.BeginSegment(1,0);
    stream.AcceptPcm16(pcm.data(),pcm.size());
    stream.EndSegment();
    for(const auto&e:stream.ReadEvents())std::cout<<e.keyword_id<<" "<<e.confidence<<"\n";
  }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}
}
