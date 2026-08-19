cmake_minimum_required(VERSION 3.18)
project(wuw_sdk VERSION 0.0.2 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(WUWSDK_BUILD_TOOLS "Build command-line tools" ON)
option(WUWSDK_BUILD_TESTS "Build SDK tests" ON)
option(WUWSDK_ENABLE_SANITIZERS "Enable address/undefined sanitizers" OFF)

set(WUWSDK_ONNXRUNTIME_ROOT "" CACHE PATH
    "ONNX Runtime root containing include/ and lib/")
set(WUWSDK_REQUIRED_ORT_VERSION "1.25.1" CACHE STRING
    "Required ONNX Runtime version (supported: 1.16.3, 1.25.1)")
set_property(CACHE WUWSDK_REQUIRED_ORT_VERSION PROPERTY STRINGS
  "1.16.3" "1.25.1")
option(WUWSDK_ALLOW_UNSUPPORTED_ORT_VERSION
       "Allow an unsupported ONNX Runtime version for local experiments" OFF)
set(ONNXRuntime_INCLUDE_DIR "" CACHE PATH "ONNX Runtime include directory")
set(ONNXRuntime_LIBRARY "" CACHE FILEPATH "ONNX Runtime shared library")

set(_supported_ort_versions "1.16.3" "1.25.1")
if(NOT WUWSDK_REQUIRED_ORT_VERSION IN_LIST _supported_ort_versions AND
   NOT WUWSDK_ALLOW_UNSUPPORTED_ORT_VERSION)
  message(FATAL_ERROR
    "Unsupported ONNX Runtime version: ${WUWSDK_REQUIRED_ORT_VERSION}. "
    "Supported release versions are 1.16.3 and 1.25.1.")
endif()
if(WUWSDK_ONNXRUNTIME_ROOT)
  set(ONNXRuntime_INCLUDE_DIR "${WUWSDK_ONNXRUNTIME_ROOT}/include")
  set(ONNXRuntime_LIBRARY
      "${WUWSDK_ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.${WUWSDK_REQUIRED_ORT_VERSION}")
endif()
if(NOT EXISTS "${ONNXRuntime_INCLUDE_DIR}/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "Set ONNXRuntime_INCLUDE_DIR to the directory containing onnxruntime_cxx_api.h")
endif()
if(NOT EXISTS "${ONNXRuntime_LIBRARY}")
  message(FATAL_ERROR
    "Set WUWSDK_ONNXRUNTIME_ROOT or ONNXRuntime_LIBRARY to the exact "
    "libonnxruntime.so.${WUWSDK_REQUIRED_ORT_VERSION}")
endif()
file(STRINGS "${ONNXRuntime_INCLUDE_DIR}/onnxruntime_c_api.h" _ort_api_line
     REGEX "^#define ORT_API_VERSION [0-9]+$")
if(WUWSDK_REQUIRED_ORT_VERSION STREQUAL "1.16.3")
  set(_expected_ort_api 16)
elseif(WUWSDK_REQUIRED_ORT_VERSION STREQUAL "1.25.1")
  set(_expected_ort_api 25)
else()
  set(_expected_ort_api "")
endif()
if(_expected_ort_api AND
   NOT _ort_api_line STREQUAL "#define ORT_API_VERSION ${_expected_ort_api}")
  message(FATAL_ERROR
    "ONNX Runtime header mismatch: ${WUWSDK_REQUIRED_ORT_VERSION} requires "
    "API ${_expected_ort_api}, found '${_ort_api_line}'")
endif()
get_filename_component(_ort_library_real "${ONNXRuntime_LIBRARY}" REALPATH)
get_filename_component(_ort_library_name "${_ort_library_real}" NAME)
if(NOT _ort_library_name STREQUAL
   "libonnxruntime.so.${WUWSDK_REQUIRED_ORT_VERSION}")
  message(FATAL_ERROR
    "ONNX Runtime library mismatch: expected libonnxruntime.so."
    "${WUWSDK_REQUIRED_ORT_VERSION}, found ${_ort_library_name}")
endif()

add_library(wuw_onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(wuw_onnxruntime PROPERTIES
  IMPORTED_LOCATION "${_ort_library_real}"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIR}")

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
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN")
target_include_directories(wuw_sdk
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${ONNXRuntime_INCLUDE_DIR})
target_link_libraries(wuw_sdk PRIVATE wuw_onnxruntime)
target_compile_options(wuw_sdk PRIVATE -Wall -Wextra -Wpedantic)

if(WUWSDK_ENABLE_SANITIZERS AND NOT MSVC)
  target_compile_options(wuw_sdk PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(wuw_sdk PRIVATE -fsanitize=address,undefined)
endif()

if(WUWSDK_BUILD_TOOLS)
  set(WUWSDK_TOOL_TARGETS)
  foreach(tool wuw_inspect wuw_stream_wav wuw_eval_manifest wuw_merge_results)
    add_executable(${tool} tools/${tool}.cc)
    list(APPEND WUWSDK_TOOL_TARGETS ${tool})
    target_include_directories(${tool} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(${tool} PRIVATE wuw_sdk wuw_onnxruntime)
    set_target_properties(${tool} PROPERTIES
      BUILD_RPATH "$ORIGIN/../lib;$ORIGIN"
      INSTALL_RPATH "$ORIGIN/../lib")
  endforeach()
endif()

if(WUWSDK_BUILD_TESTS)
  enable_testing()
  add_executable(test_config tests/test_config.cc src/config.cc src/util.cc)
  add_executable(test_scoring tests/test_scoring.cc src/scoring.cc)
  foreach(test_name test_config test_scoring)
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include ${ONNXRuntime_INCLUDE_DIR})
    target_link_libraries(${test_name} PRIVATE wuw_onnxruntime)
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
  add_executable(test_stream_api tests/test_stream_api.cc)
  target_link_libraries(test_stream_api PRIVATE wuw_sdk wuw_onnxruntime)
  add_test(NAME test_stream_api COMMAND test_stream_api)
endif()

include(GNUInstallDirs)
install(TARGETS wuw_sdk LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
if(WUWSDK_BUILD_TOOLS)
  install(TARGETS ${WUWSDK_TOOL_TARGETS} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
