#include <cassert>
#include <iostream>
#include "internal.h"
int main(){auto j=wuw::ParseJson("{\"x\":2,\"a\":[true,\"ok\"]}");assert(j.at("x").number()==2);assert(j.at("a").array()[0].boolean());assert(j.at("a").array()[1].string()=="ok");bool failed=false;try{wuw::ParseJson("{\"x\":1,}");}catch(const wuw::Error&){failed=true;}assert(failed);std::cout<<"ok\n";}
