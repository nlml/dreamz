cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dcgan)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)

find_package(RTMIDI 2.1.0)
if(RTMIDI_FOUND)
    message(STATUS "Enabled MIDI support (ALSA/RtMidi) ye")
else()
    message("MIDI support disabled (RtMidi unavailable)")
endif()

find_package(Torch REQUIRED)
find_package(SDL2 REQUIRED)
find_package(RTMIDI REQUIRED)

add_executable(dcgan dcgan.cpp)

INCLUDE(FindPkgConfig)

PKG_SEARCH_MODULE(RTMIDI REQUIRED rtmidi)
PKG_SEARCH_MODULE(SDL2 REQUIRED sdl2)

include_directories(${SDL2_INCLUDE_DIRS})
target_link_libraries(dcgan ${SDL2_LIBRARIES} ${TORCH_LIBRARIES})
set_property(TARGET dcgan PROPERTY CXX_STANDARD 11)

