#include <cassert>
#include <cstring>
#include <iostream>
#include "wuw_sdk/c_api.h"
int main(){WuwSdkEngineOptions o;wuw_sdk_engine_options_init(&o);assert(o.abi_version==WUW_SDK_ABI_VERSION);assert(o.struct_size==sizeof(o));WuwSdkEngine*e=reinterpret_cast<WuwSdkEngine*>(1);auto s=wuw_sdk_engine_create("",&o,&e);assert(s==WUW_SDK_STATUS_INVALID_ARGUMENT);assert(e==nullptr);assert(std::strlen(wuw_sdk_last_error_message(nullptr))>0);std::cout<<"ok\n";}
