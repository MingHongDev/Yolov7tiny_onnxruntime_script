cmake_minimum_required(VERSION 3.10)
project(OnnxRuntimeScript)

set(CMAKE_CXX_STANDARD 17)
find_package(onnxruntime REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNXRUNTIME_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_LIST_DIR}/src)
add_executable(OnnxRuntimeScript src/main.cpp src/OnnxInferenceBase.cpp src/Yolov7.cpp)
target_link_libraries(OnnxRuntimeScript PRIVATE onnxruntime::onnxruntime ${OpenCV_LIBRARIES})
