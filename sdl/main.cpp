#include <SDL2/SDL.h>
#include <SDL2/SDL_render.h>
#include <iostream>
#include <vector>

using namespace std;

int main( int argc, char** argv )
{
    SDL_Init( SDL_INIT_EVERYTHING );
    atexit( SDL_Quit );

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
    SDL_GetRendererInfo( renderer, &info );
    cout << "Renderer name: " << info.name << endl;
    cout << "Texture formats: " << endl;
    for( Uint32 i = 0; i < info.num_texture_formats; i++ )
    {
        cout << SDL_GetPixelFormatName( info.texture_formats[i] ) << endl;
    }

    const unsigned int texWidth = 1024;
    const unsigned int texHeight = 1024;
    SDL_Texture* texture = SDL_CreateTexture
        (
        renderer,
        SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING,
        texWidth, texHeight
        );

    vector< unsigned char > pixels( texWidth * texHeight * 4, 0 );

    SDL_Event event;
    bool running = true;
    while( running )
    {
        const Uint64 start = SDL_GetPerformanceCounter();

        SDL_SetRenderDrawColor( renderer, 0, 0, 0, SDL_ALPHA_OPAQUE );
        SDL_RenderClear( renderer );

        while( SDL_PollEvent( &event ) )
        {
            if( ( SDL_QUIT == event.type ) ||
                ( SDL_KEYDOWN == event.type && SDL_SCANCODE_ESCAPE == event.key.keysym.scancode ) )
            {
                running = false;
                break;
            }
        }

        // splat down some random pixels
        const unsigned int r = rand() % 256;
        const unsigned int g = rand() % 256;
        const unsigned int b = rand() % 256;
        for( unsigned int x = 0; x < texWidth; x++ )
        {
            for( unsigned int y = 0; y < texHeight; y++ )
            {
                const unsigned int offset = (x * texWidth + y) * 4;
                pixels[ offset + 0 ] = b;        // b
                pixels[ offset + 1 ] = g;        // g
                pixels[ offset + 2 ] = r;        // r
                pixels[ offset + 3 ] = SDL_ALPHA_OPAQUE;    // a
            }
        }

        SDL_UpdateTexture
            (
            texture,
            NULL,
            &pixels[0],
            texWidth * 4
            );

        SDL_RenderCopy( renderer, texture, NULL, NULL );
        SDL_RenderPresent( renderer );

        const Uint64 end = SDL_GetPerformanceCounter();
        const static Uint64 freq = SDL_GetPerformanceFrequency();
        const double seconds = ( end - start ) / static_cast< double >( freq );
        cout << "Frame time: " << seconds * 1000.0 << "ms" << endl;
    }

    SDL_DestroyRenderer( renderer );
    SDL_DestroyWindow( window );
    SDL_Quit();
}
