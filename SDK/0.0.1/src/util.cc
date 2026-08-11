#include "internal.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace wuw {
namespace {
constexpr std::array<uint32_t,64> K={0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
uint32_t R(uint32_t x,int n){return (x>>n)|(x<<(32-n));}
uint32_t B(const unsigned char* p){return(uint32_t(p[0])<<24)|(uint32_t(p[1])<<16)|(uint32_t(p[2])<<8)|p[3];}
void Block(const unsigned char* p,std::array<uint32_t,8>* s){std::array<uint32_t,64>w{};for(int i=0;i<16;++i)w[i]=B(p+4*i);for(int i=16;i<64;++i){uint32_t a=R(w[i-15],7)^R(w[i-15],18)^(w[i-15]>>3),b=R(w[i-2],17)^R(w[i-2],19)^(w[i-2]>>10);w[i]=w[i-16]+a+w[i-7]+b;}uint32_t a=(*s)[0],b=(*s)[1],c=(*s)[2],d=(*s)[3],e=(*s)[4],f=(*s)[5],g=(*s)[6],h=(*s)[7];for(int i=0;i<64;++i){uint32_t s1=R(e,6)^R(e,11)^R(e,25),ch=(e&f)^((~e)&g),t1=h+s1+ch+K[i]+w[i],s0=R(a,2)^R(a,13)^R(a,22),maj=(a&b)^(a&c)^(b&c),t2=s0+maj;h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}(*s)[0]+=a;(*s)[1]+=b;(*s)[2]+=c;(*s)[3]+=d;(*s)[4]+=e;(*s)[5]+=f;(*s)[6]+=g;(*s)[7]+=h;}
std::string Trim(std::string s){auto n=[](unsigned char c){return!std::isspace(c);};s.erase(s.begin(),std::find_if(s.begin(),s.end(),n));s.erase(std::find_if(s.rbegin(),s.rend(),n).base(),s.end());return s;}
}
std::string Sha256File(const std::filesystem::path& path){std::ifstream in(path,std::ios::binary);if(!in)throw Error(WUW_SDK_STATUS_NOT_FOUND,"cannot checksum "+path.string());std::array<uint32_t,8>s={0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};std::array<unsigned char,64>b{};uint64_t total=0;while(in){in.read(reinterpret_cast<char*>(b.data()),64);size_t n=static_cast<size_t>(in.gcount());if(n==64){Block(b.data(),&s);total+=n;continue;}total+=n;b[n]=0x80;std::fill(b.begin()+n+1,b.end(),0);if(n>=56){Block(b.data(),&s);b.fill(0);}uint64_t bits=total*8;for(int i=0;i<8;++i)b[63-i]=static_cast<unsigned char>((bits>>(8*i))&255);Block(b.data(),&s);break;}std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto v:s)out<<std::setw(8)<<v;return out.str();}
void VerifyChecksums(const std::filesystem::path& dir){auto path=dir/"SHA256SUMS";std::ifstream in(path);if(!in)throw Error(WUW_SDK_STATUS_NOT_FOUND,"missing "+path.string());std::string line;int no=0;while(std::getline(in,line)){++no;line=Trim(line);if(line.empty()||line[0]=='#')continue;std::istringstream ss(line);std::string expected,name;ss>>expected;std::getline(ss,name);name=Trim(name);if(!name.empty()&&name[0]=='*')name.erase(name.begin());if(expected.size()!=64||name.empty()||name.find("..")!=std::string::npos||std::filesystem::path(name).is_absolute())throw Error(WUW_SDK_STATUS_INCOMPATIBLE_MODEL,"invalid SHA256SUMS line "+std::to_string(no));auto file=dir/name;if(Sha256File(file)!=expected)throw Error(WUW_SDK_STATUS_INCOMPATIBLE_MODEL,"checksum mismatch: "+name);}}
void CopyString(const std::string& source,char* destination,size_t capacity){if(!destination||capacity==0)return;size_t n=std::min(source.size(),capacity-1);std::memcpy(destination,source.data(),n);destination[n]='\0';}
}  // namespace wuw
