#ifndef WUW_SDK_TOOL_COMMON_H_
#define WUW_SDK_TOOL_COMMON_H_
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "wuw_sdk/wuw.hpp"
namespace tool {
inline uint16_t U16(const unsigned char*p){return uint16_t(p[0])|(uint16_t(p[1])<<8);}inline uint32_t U32(const unsigned char*p){return uint32_t(p[0])|(uint32_t(p[1])<<8)|(uint32_t(p[2])<<16)|(uint32_t(p[3])<<24);}
inline std::vector<int16_t> ReadWav(const std::string&path){std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open WAV: "+path);unsigned char head[12];in.read(reinterpret_cast<char*>(head),12);if(in.gcount()!=12||std::string(reinterpret_cast<char*>(head),4)!="RIFF"||std::string(reinterpret_cast<char*>(head+8),4)!="WAVE")throw std::runtime_error("not a RIFF/WAVE file: "+path);uint16_t format=0,channels=0,bits=0;uint32_t rate=0;std::vector<char>data;while(in){unsigned char h[8];in.read(reinterpret_cast<char*>(h),8);if(in.gcount()!=8)break;uint32_t n=U32(h+4);std::string id(reinterpret_cast<char*>(h),4);std::vector<char>chunk(n);in.read(chunk.data(),n);if(static_cast<uint32_t>(in.gcount())!=n)throw std::runtime_error("truncated WAV: "+path);if(n&1)in.get();if(id=="fmt "&&n>=16){auto*p=reinterpret_cast<unsigned char*>(chunk.data());format=U16(p);channels=U16(p+2);rate=U32(p+4);bits=U16(p+14);}else if(id=="data")data=std::move(chunk);}if(format!=1||channels!=1||rate!=16000||bits!=16||data.size()%2)throw std::runtime_error("SDK requires mono PCM16 16000 Hz WAV: "+path);std::vector<int16_t>out(data.size()/2);for(size_t i=0;i<out.size();++i){auto*p=reinterpret_cast<const unsigned char*>(data.data()+2*i);out[i]=static_cast<int16_t>(U16(p));}return out;}
inline std::string JsonString(const std::string&line,const std::string&key){std::string needle="\""+key+"\"";size_t p=line.find(needle);if(p==std::string::npos)return{};p=line.find(':',p+needle.size());if(p==std::string::npos)return{};p=line.find('"',p+1);if(p==std::string::npos)return{};std::string out;for(++p;p<line.size();++p){char c=line[p];if(c=='"')return out;if(c=='\\'&&p+1<line.size()){char e=line[++p];if(e=='n')out+='\n';else if(e=='t')out+='\t';else out+=e;}else out+=c;}return{};}
inline std::string Escape(const std::string&s){std::string o;for(char c:s){if(c=='"'||c=='\\'){o+='\\';o+=c;}else if(c=='\n')o+="\\n";else o+=c;}return o;}
inline void Check(WuwSdkStatus s,const void*h=nullptr){if(s!=WUW_SDK_STATUS_OK)throw std::runtime_error(wuw_sdk_last_error_message(h));}
}
#endif
