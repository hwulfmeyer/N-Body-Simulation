# CMakeLists.txt
cmake_minimum_required(VERSION 2.8)
project(n-bodysim)

# set( CUDA_TOOLKIT_ROOT_DIR "/Developer/NVIDIA/CUDA-8.0/")
find_package(CUDA QUIET REQUIRED)
find_package(OpenGL REQUIRED)

set( CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set( ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set( SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set( SFML_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include/SFML/)
set( GLEW_PATH ${CMAKE_CURRENT_SOURCE_DIR}/include/glew/)

set(SOURCE
   ${SRC_DIR}/main.cpp
   ${SRC_DIR}/body.cpp
   ${SRC_DIR}/cpu_computing.cpp
)

set(CUDA_SOURCE
   ${SOURCE}
   ${SRC_DIR}/cuda_computing.cu
)

set(INCLUDE_DIRS
   ${CMAKE_CURRENT_SOURCE_DIR}/include
   ${SFML_PATH}/include
   ${GLEW_PATH}/include/GL
)

set(LINK_DIRS
   ${SFML_PATH}/lib
   ${GLEW_PATH}/lib
)

IF (WIN32)
ELSE()
  set( CMAKE_CXX_FLAGS "-g -W -Wall -std=c++11")
ENDIF()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

set(CUDA_NVCC_FLAGS "-Wno-deprecated-gpu-targets --expt-relaxed-constexpr --resource-usage -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52")

include_directories( ${INCLUDE_DIRS})
add_definitions(-DGLEW_STATIC)
link_directories(${LINK_DIRS})

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build)
add_executable(nbodysim ${SOURCE})
cuda_add_executable(nbodysim_device ${CUDA_SOURCE} )
set_target_properties(nbodysim_device PROPERTIES COMPILE_FLAGS "-DDEVICE_COMPILE")

target_link_libraries(nbodysim ${OPENGL_LIBRARY} libglew32)
target_link_libraries(nbodysim_device ${OPENGL_LIBRARY} libglew32)

# Detect and add SFML
set(CMAKE_MODULE_PATH ${SFML_PATH}/cmake/Modules)
set(SFML_ROOT ${SFML_PATH})
set(SFML_STATIC_LIBRARIES TRUE)
find_package(SFML 2 REQUIRED system window graphics network audio)
IF (SFML_FOUND)
	target_link_libraries(nbodysim ${SFML_LIBRARIES})
	target_link_libraries(nbodysim ${SFML_DEPENDENCIES})
	target_link_libraries(nbodysim_device ${SFML_LIBRARIES})
	target_link_libraries(nbodysim_device ${SFML_DEPENDENCIES})
ELSE()
    message(SEND_ERROR "SFML library not found. Please set SFML_ROOT.")
ENDIF()
 