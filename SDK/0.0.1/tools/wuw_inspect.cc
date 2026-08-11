#include <iostream>
#include "wuw_sdk/wuw.hpp"
int main(int argc,char**argv){try{if(argc!=2){std::cerr<<"usage: wuw_inspect MODEL_DIR\n";return 2;}wuw_sdk::Engine e(argv[1]);std::cout<<"SDK "<<wuw_sdk_version()<<", ABI "<<wuw_sdk_abi_version()<<"\nstage2_threshold "<<e.stage2_threshold()<<"\n";for(size_t i=0;i<e.keyword_count();++i){auto k=e.keyword(i);std::cout<<k.keyword_id<<"\t"<<k.display_text<<"\t"<<k.stage1_threshold<<"\n";}return 0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
