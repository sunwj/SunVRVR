cmake_minimum_required(VERSION 3.3)
project(SunVRVR)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_INCLUDE_CURRENT_DIR ON)

#Qt
find_package(Qt4 REQUIRED)
INCLUDE(${QT_USE_FILE})
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(RESOURCES_FILES qdarkstyle/style.qrc)

file(GLOB HOST_SOURCES *.cpp)

add_executable(SunVRVR ${HOST_SOURCES} ${RESOURCES_FILES})
target_link_libraries(SunVRVR Qt4::QtCore Qt4::QtGui Qt4::QtOpenGL)