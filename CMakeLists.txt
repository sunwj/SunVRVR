cmake_minimum_required(VERSION 3.3)
project(SunVRVR)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# CUDA settings
find_package(CUDA QUIET REQUIRED)
list(APPEND CUDA_NVCC_FLAGS --compiler-options -fno-strict-aliasing -use_fast_math -Xptxas -v)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_30,code=sm_30)
set(CUDA_INCLUDE_DIRS /usr/local/cuda/include/)
include_directories(${CUDA_INCLUDE_DIRS})

#Qt
find_package(Qt4 COMPONENTS QtCore QtGui QtOpenGL REQUIRED)
include(${QT_USE_FILE})
set(QT_USE_QTOPENGL TRUE)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(RESOURCES_FILES qdarkstyle/style.qrc pathtracer.cu)

#VTK
set(VTK_DIR /Volumes/Free/SDK/CTK/build/VTK-build)
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

#CTK
set(CTK_DIR /Volumes/Free/SDK/CTK/build/)
#set(CTK_DIR /Users/sunwj/Desktop/SDK/CTK/build)
find_package(CTK REQUIRED)
include(${CTK_USE_FILE})
set(CTK_LIBRARIES CTKCore CTKWidgets CTKVisualizationVTKCore CTKVisualizationVTKWidgets)

file(GLOB HOST_SOURCES *.cpp)
file(GLOB DEVICE_SOURCES *.cu)

cuda_compile(DEVICE_OBJS ${DEVICE_SOURCES})

add_executable(SunVRVR ${HOST_SOURCES} ${RESOURCES_FILES} cuda_utils.h canvas.cpp canvas.h pathtracer.h)
target_link_libraries(SunVRVR ${DEVICE_OBJS} ${QT_LIBRARIES} ${VTK_LIBRARIES} ${CTK_LIBRARIES} ${CUDA_LIBRARIES})
