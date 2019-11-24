cmake_minimum_required(VERSION 3.5)
project(RicciFlow)

set(SOURCE_FILES raytrace.cpp framework.cpp)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-long-long -Wreturn-type -pedantic")


find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenMP)
include_directories(${OPENGL_INCLUDE_DIRS} ${GLUT_INCLUDE_DIRS} ${GLEW_INCLUDE_DIRS})

#link_libraries(${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARIES} OpenMP::OpenMP_CXX)
link_libraries(${OPENGL_LIBRARIES} ${GLUT_LIBRARY} ${GLEW_LIBRARIES})

add_executable(Ketres ${SOURCE_FILES})
