#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "tool_common.h"
int main(int argc,char**argv){try{if(argc<4){std::cerr<<"usage: wuw_merge_results OUTPUT.jsonl SUMMARY.json SHARD...\n";return 2;}std::vector<std::pair<uint64_t,std::string>>rows;uint64_t errors=0,detected=0;for(int a=3;a<argc;++a){std::ifstream in(argv[a]);if(!in)throw std::runtime_error(std::string("missing shard: ")+argv[a]);std::string line;while(std::getline(in,line)){size_t p=line.find("\"source_index\":");if(p==std::string::npos)continue;uint64_t i=std::stoull(line.substr(p+15));if(line.find("\"error\"")!=std::string::npos)++errors;if(line.find("\"events\":[]")==std::string::npos&&line.find("\"events\"")!=std::string::npos)++detected;rows.push_back({i,line});}}std::sort(rows.begin(),rows.end(),[](const auto&a,const auto&b){return a.first<b.first;});for(size_t i=1;i<rows.size();++i)if(rows[i-1].first==rows[i].first)throw std::runtime_error("duplicate source_index");std::ofstream merged(argv[1]);for(auto&r:rows)merged<<r.second<<'\n';std::ofstream summary(argv[2]);summary<<"{\n  \"records\": "<<rows.size()<<",\n  \"records_with_detection\": "<<detected<<",\n  \"errors\": "<<errors<<"\n}\n";std::cout<<"merged "<<rows.size()<<" records\n";return errors?1:0;}catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
