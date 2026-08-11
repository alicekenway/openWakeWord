cmake_minimum_required(VERSION 3.18)
project(wuw_sdk VERSION 0.0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(WUWSDK_BUILD_TOOLS "Build command-line tools" ON)
option(WUWSDK_BUILD_TESTS "Build SDK tests" ON)
option(WUWSDK_ENABLE_SANITIZERS "Enable address/undefined sanitizers" OFF)

set(ONNXRuntime_INCLUDE_DIR "" CACHE PATH "ONNX Runtime include directory")
set(ONNXRuntime_LIBRARY "" CACHE FILEPATH "ONNX Runtime shared library")
if(NOT EXISTS "${ONNXRuntime_INCLUDE_DIR}/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "Set ONNXRuntime_INCLUDE_DIR to the directory containing onnxruntime_cxx_api.h")
endif()
if(NOT EXISTS "${ONNXRuntime_LIBRARY}")
  message(FATAL_ERROR "Set ONNXRuntime_LIBRARY to libonnxruntime.so")
endif()

add_library(wuw_sdk SHARED
  src/c_api.cc
  src/config.cc
  src/fbank.cc
  src/model.cc
  src/scoring.cc
  src/stream.cc
  src/util.cc)
set_target_properties(wuw_sdk PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  BUILD_RPATH "$ORIGIN")
target_include_directories(wuw_sdk
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${ONNXRuntime_INCLUDE_DIR})
target_link_libraries(wuw_sdk PRIVATE ${ONNXRuntime_LIBRARY})
target_compile_options(wuw_sdk PRIVATE -Wall -Wextra -Wpedantic)

if(WUWSDK_ENABLE_SANITIZERS AND NOT MSVC)
  target_compile_options(wuw_sdk PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(wuw_sdk PRIVATE -fsanitize=address,undefined)
endif()

if(WUWSDK_BUILD_TOOLS)
  foreach(tool wuw_inspect wuw_stream_wav wuw_eval_manifest wuw_merge_results)
    add_executable(${tool} tools/${tool}.cc)
    target_include_directories(${tool} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(${tool} PRIVATE wuw_sdk ${ONNXRuntime_LIBRARY})
    set_target_properties(${tool} PROPERTIES BUILD_RPATH "$ORIGIN/../lib;$ORIGIN")
  endforeach()
endif()

if(WUWSDK_BUILD_TESTS)
  enable_testing()
  add_executable(test_config tests/test_config.cc src/config.cc src/util.cc)
  add_executable(test_scoring tests/test_scoring.cc src/scoring.cc)
  foreach(test_name test_config test_scoring)
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include ${ONNXRuntime_INCLUDE_DIR})
    target_link_libraries(${test_name} PRIVATE ${ONNXRuntime_LIBRARY})
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
  add_executable(test_stream_api tests/test_stream_api.cc)
  target_link_libraries(test_stream_api PRIVATE wuw_sdk ${ONNXRuntime_LIBRARY})
  add_test(NAME test_stream_api COMMAND test_stream_api)
endif()

include(GNUInstallDirs)
install(TARGETS wuw_sdk LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
