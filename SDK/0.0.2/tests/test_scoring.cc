#include <cassert>
#include <cmath>
#include <iostream>
#include "internal.h"
int main(){wuw::BoundedCtcScorer scorer({1,2},0,8);const float frames[][3]={{-2,-.1f,-3},{-.1f,-2,-3},{-2,-3,-.1f},{-.1f,-3,-2}};wuw::Trace t;for(int i=0;i<4;++i)t=scorer.Step(frames[i],3,i);assert(t.start==0);assert(t.end==2);assert(std::isfinite(t.score));std::vector<float>p;for(auto&f:frames)p.insert(p.end(),f,f+3);float confidence=wuw::KeywordVsFillerConfidence(p,4,3,{1,2},0,4,2);assert(confidence>0.5f&&confidence<=1.0f);std::cout<<"confidence="<<confidence<<"\n";}
