#include <torch/torch.h>
#include <iostream>
#include <time.h>
#include <chrono>
#include <thread>
#include <SDL2/SDL.h>
#include <SDL2/SDL_render.h>


#define WIDTH 600
#define HEIGHT 600


int main(int argc, char ** argv)
{
    bool leftMouseButtonDown = false;
    bool quit = false;
    SDL_Event event;

    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window * window = SDL_CreateWindow("SDL2 Pixel Drawing",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WIDTH, HEIGHT, 0);

    SDL_Renderer * renderer = SDL_CreateRenderer(window, -1, 0);
    SDL_Texture * texture = SDL_CreateTexture(renderer,
        SDL_PIXELFORMAT_RGB888, SDL_TEXTUREACCESS_STATIC, WIDTH, HEIGHT);
    Uint8 * pixels = new Uint8[WIDTH * HEIGHT * 4];

    memset(pixels, 255, WIDTH * HEIGHT * 4 * sizeof(Uint8));

    while (!quit)
    {
        clock_t tStart = clock();
        SDL_UpdateTexture(texture, NULL, pixels, WIDTH * sizeof(Uint32));

        torch::Tensor t = torch::rand({WIDTH, HEIGHT, 4});
        t = t * 255.0;
        t = at::reshape(t, -1);
        t = t.toType(torch::kUInt8);
        // std::cout << "tensor dtype = " << t.dtype() << std::endl;
        auto array = t.accessor<unsigned char, 1>();
        
        for(int i = 0; i < array.size(0); i++)
        {
            pixels[i] = array[i];
        }

        // SDL_WaitEvent(&event);

        // switch (event.type)
        // {
        //     case SDL_QUIT:
        //         quit = true;
        //         break;
        //     case SDL_MOUSEBUTTONUP:
        //         if (event.button.button == SDL_BUTTON_LEFT)
        //             leftMouseButtonDown = false;
        //         break;
        //     case SDL_MOUSEBUTTONDOWN:
        //         if (event.button.button == SDL_BUTTON_LEFT)
        //             leftMouseButtonDown = true;
        //     case SDL_MOUSEMOTION:
        //         if (leftMouseButtonDown)
        //         {
        //             int mouseX = event.motion.x;
        //             int mouseY = event.motion.y;
        //             pixels[(mouseY * WIDTH + mouseX) * 4 + 2] = 255;
        //             pixels[(mouseY * WIDTH + 1 + mouseX) * 4 + 2] = 255;
        //             pixels[(mouseY * WIDTH - 1 + mouseX) * 4 + 2] = 255;
        //         }
        //         break;
        // }

        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
        printf("Time taken: %.6fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);
    }

    delete[] pixels;
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);

    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
