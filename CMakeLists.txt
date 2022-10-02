cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(libtorch_test)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(mnist mnist.cpp)
target_link_libraries(mnist "${TORCH_LIBRARIES}")
set_property(TARGET mnist PROPERTY CXX_STANDARD 17)
