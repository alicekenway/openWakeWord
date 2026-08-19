#include "internal.h"

#include <algorithm>
#include <cmath>
#include <complex>

namespace wuw {
namespace {
constexpr double kPi = 3.14159265358979323846;
float Mel(float hz) { return 1127.0f * std::log1p(hz / 700.0f); }
void Fft(std::vector<std::complex<float>>* values) {
  auto& a=*values; const size_t n=a.size();
  for(size_t i=1,j=0;i<n;++i){size_t bit=n>>1;for(;j&bit;bit>>=1)j^=bit;j^=bit;if(i<j)std::swap(a[i],a[j]);}
  for(size_t len=2;len<=n;len<<=1){float angle=static_cast<float>(-2*kPi/len);std::complex<float>wlen(std::cos(angle),std::sin(angle));for(size_t i=0;i<n;i+=len){std::complex<float>w(1,0);for(size_t j=0;j<len/2;++j){auto u=a[i+j],v=a[i+j+len/2]*w;a[i+j]=u+v;a[i+j+len/2]=u-v;w*=wlen;}}}
}
}

Fbank::Fbank(int sample_rate,int feature_dim,int frame_length,int frame_shift)
    :sample_rate_(sample_rate),feature_dim_(feature_dim),frame_length_(frame_length),frame_shift_(frame_shift){
  window_.resize(frame_length_);for(int i=0;i<frame_length_;++i)window_[i]=std::pow(0.5f-0.5f*std::cos(static_cast<float>(2*kPi*i/(frame_length_-1))),0.85f);
  const int fft=512,bins=fft/2;float width=static_cast<float>(sample_rate_)/fft,lo=Mel(20.0f),hi=Mel(sample_rate_/2.0f),delta=(hi-lo)/(feature_dim_+1);
  mel_bins_.resize(feature_dim_);
  for(int b=0;b<feature_dim_;++b){float l=lo+b*delta,c=lo+(b+1)*delta,r=lo+(b+2)*delta;for(int i=0;i<bins;++i){float m=Mel(width*i);if(m>l&&m<r){float w=m<=c?(m-l)/(c-l):(r-m)/(r-c);mel_bins_[b].push_back({i,w});}}if(mel_bins_[b].empty())throw Error(WUW_SDK_STATUS_INTERNAL_ERROR,"empty mel filter");}
}
void Fbank::Reset(){samples_.clear();ready_.clear();next_frame_start_=0;}
void Fbank::Accept(const int16_t* samples,size_t count){if(count)samples_.insert(samples_.end(),samples,samples+count);while(next_frame_start_+static_cast<size_t>(frame_length_)<=samples_.size()){ready_.push_back(Compute(samples_.data()+next_frame_start_));next_frame_start_+=frame_shift_;}if(next_frame_start_>8192){samples_.erase(samples_.begin(),samples_.begin()+static_cast<std::ptrdiff_t>(next_frame_start_));next_frame_start_=0;}}
std::vector<std::vector<float>> Fbank::ReadFrames(){std::vector<std::vector<float>> out;out.swap(ready_);return out;}
std::vector<float> Fbank::Compute(const int16_t* frame)const{
  std::vector<std::complex<float>> fft(512);float mean=0;for(int i=0;i<frame_length_;++i)mean+=frame[i];mean/=frame_length_;
  std::vector<float>x(frame_length_);for(int i=0;i<frame_length_;++i)x[i]=static_cast<float>(frame[i])-mean;for(int i=frame_length_-1;i>0;--i)x[i]-=0.97f*x[i-1];x[0]-=0.97f*x[0];for(int i=0;i<frame_length_;++i)fft[i]=x[i]*window_[i];Fft(&fft);
  std::vector<float> power(256);for(int i=0;i<256;++i)power[i]=std::norm(fft[i]);std::vector<float> out(feature_dim_);for(int b=0;b<feature_dim_;++b){float e=0;for(auto [i,w]:mel_bins_[b])e+=power[i]*w;out[b]=std::log(std::max(e,std::numeric_limits<float>::epsilon()));}return out;
}
}  // namespace wuw
