#include <SDL2/SDL.h>
#include <SDL2/SDL_render.h>
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

namespace py = pybind11;
using namespace std;


SDL_Window * window;
SDL_Renderer * renderer;
SDL_RendererInfo info;
SDL_Texture * texture;

// Numpy matrix classes
py::array_t < unsigned char > make_array(const py::ssize_t size) {
    // No pointer is passed, so NumPy will allocate the buffer
    return py::array_t < unsigned char > (size);
}

void render(unsigned char * ptr, int width) {
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
    SDL_RenderClear(renderer);
    SDL_UpdateTexture(texture, NULL, &ptr[0], width * 3);
    SDL_RenderCopy(renderer, texture, NULL, NULL);
    SDL_RenderPresent(renderer);
}

// wrap C++ function with NumPy array IO
void wrapper(py::array_t < unsigned char > array, int width) {
    // check input dimensions
    if (array.ndim() != 1)
        throw std::runtime_error("Input should be 1D NumPy array");
    auto buf = array.request();
    unsigned char * ptr = (unsigned char * ) buf.ptr;
    // call pure C++ function
    render(ptr, width);
}

void setup(int w, int h, int win_x, int win_y) {
    window = SDL_CreateWindow(
        "SDL2",
        win_y, win_x,
        // SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        h, w,
        // SDL_WINDOW_SHOWN
        SDL_WINDOW_FULLSCREEN_DESKTOP
    );

    renderer = SDL_CreateRenderer(
        window,
        -1,
        SDL_RENDERER_ACCELERATED
    );

    texture = SDL_CreateTexture(renderer,
        SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, h, w);
}

void killMe() {
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

PYBIND11_MODULE(example, m) {
    m.doc() = "render pixels to SDL2 window"; // optional module docstring
    m.def("setup", & setup, "Start renderer", py::arg("h"), py::arg("w"), py::arg("win_y"), py::arg("win_x"));
    m.def("render", & wrapper, "Render one frame");
    m.def("kill", & killMe, "Shut down");
    m.def("make_array", & make_array,
        py::return_value_policy::move); // Return policy can be left default, i.e. return_value_policy::automatic
}