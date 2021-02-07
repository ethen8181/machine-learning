cmake_minimum_required(VERSION 3.0.0)
project(gbt_inference VERSION 0.1.0)

set(GBT_INFERENCE_HEADER_FILES
    src/xgb.hpp
    src/m2cgen.hpp)

set(GBT_INFERENCE_CPP_FILES
    src/xgb.cpp
    src/m2cgen.cpp)

add_library(gbt_inference SHARED
    ${GBT_INFERENCE_HEADER_FILES}
    ${GBT_INFERENCE_CPP_FILES}
)

add_executable(main src/main.cpp)

target_link_libraries(main PRIVATE gbt_inference)
