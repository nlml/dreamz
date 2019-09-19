cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dcgan)

find_package(Torch REQUIRED)
find_package(SDL2 REQUIRED)

add_executable(dcgan dcgan.cpp)

INCLUDE(FindPkgConfig)
PKG_SEARCH_MODULE(SDL2 REQUIRED sdl2)

include_directories(${SDL2_INCLUDE_DIRS})
target_link_libraries(dcgan ${SDL2_LIBRARIES} "${TORCH_LIBRARIES}")
set_property(TARGET dcgan PROPERTY CXX_STANDARD 11)
