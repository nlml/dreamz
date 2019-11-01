#include <SDL2/SDL.h>
#include <SDL2/SDL_render.h>
#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

namespace py = pybind11;
using namespace std;


SDL_Window* window = SDL_CreateWindow
    (
    "SDL2",
    SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
    600, 600,
    SDL_WINDOW_SHOWN
    );

SDL_Renderer* renderer = SDL_CreateRenderer
    (
    window,
    -1,
    SDL_RENDERER_ACCELERATED
    );

SDL_RendererInfo info;

const unsigned int texWidth = 1000;
const unsigned int texHeight = 1000;
// SDL_Texture* texture = SDL_CreateTexture
//     (
//     renderer,
//     SDL_PIXELFORMAT_ARGB8888,
//     SDL_TEXTUREACCESS_STREAMING,
//     texWidth, texHeight
//     );

SDL_Texture *texture = SDL_CreateTexture(renderer,
                       SDL_PIXELFORMAT_RGB888, SDL_TEXTUREACCESS_STREAMING, texWidth, texHeight);

vector< unsigned char > pixels( texWidth * texHeight * 4, 0 );

// Numpy matrix classes
py::array_t<unsigned char> make_array(const py::ssize_t size) {
    // No pointer is passed, so NumPy will allocate the buffer
    return py::array_t<unsigned char>(size);
}

void render(unsigned char* ptr){
    // const Uint64 start = SDL_GetPerformanceCounter();

    SDL_SetRenderDrawColor( renderer, 0, 0, 0, SDL_ALPHA_OPAQUE );
    SDL_RenderClear( renderer );

    // // splat down some random pixels
    // for( unsigned char i = 0; i < 1000; i++ )
    // {
    //     const unsigned char x = rand() % texWidth;
    //     const unsigned char y = rand() % texHeight;

    //     const unsigned char offset = ( texWidth * 4 * y ) + x * 4;
    //     pixels[ offset + 0 ] = rand() % 256;        // b
    //     pixels[ offset + 1 ] = rand() % 256;        // g
    //     pixels[ offset + 2 ] = rand() % 256;        // r
    //     pixels[ offset + 3 ] = SDL_ALPHA_OPAQUE;    // a
    // }

    SDL_UpdateTexture
        (
        texture,
        NULL,
        &ptr[0],
        texWidth * 4
        );

    SDL_RenderCopy( renderer, texture, NULL, NULL );
    SDL_RenderPresent( renderer );

    // const Uint64 end = SDL_GetPerformanceCounter();
    // const static Uint64 freq = SDL_GetPerformanceFrequency();
    // const double seconds = ( end - start ) / static_cast< double >( freq );
    // cout << "Frame time: " << seconds * 1000.0 << "ms" << endl;
}

// wrap C++ function with NumPy array IO
void wrapper(py::array_t<unsigned char> array) {
    // check input dimensions
    if ( array.ndim() != 1 )
        throw std::runtime_error("Input should be 1D NumPy array");

    auto buf = array.request();

    int N = array.shape()[0], M = array.shape()[1], D = array.shape()[2];

    unsigned char* ptr = (unsigned char*) buf.ptr;
    // call pure C++ function
    render(ptr);
}


void setup(int i, int j)
{
    SDL_Init( SDL_INIT_EVERYTHING );
    atexit( SDL_Quit );
    SDL_GetRendererInfo( renderer, &info );
    cout << "Renderer name: " << info.name << endl;
    cout << "Texture formats: " << endl;
    for( Uint32 i = 0; i < info.num_texture_formats; i++ )
    {
        cout << SDL_GetPixelFormatName( info.texture_formats[i] ) << endl;
    }
}

void killMe(){
    SDL_DestroyRenderer( renderer );
    SDL_DestroyWindow( window );
    SDL_Quit();
}

PYBIND11_MODULE(example, m) {
    m.doc() = "render pixels to SDL2 window"; // optional module docstring
    m.def("setup", &setup, "Start renderer", py::arg("i"), py::arg("j"));
    m.def("render", &wrapper, "Render one frame");
    m.def("kill", &killMe, "Shut down");
    m.def("make_array", &make_array,
         py::return_value_policy::move); // Return policy can be left default, i.e. return_value_policy::automatic
    // m.def("make_array", &make_array, "Functiony",
    //       py::return_value_policy::move); // Return policy can be left default, i.e. return_value_policy::automatic
}
