#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <time.h>
#include <chrono>
#include <thread>
#include <SDL2/SDL.h>
#include <SDL2/SDL_render.h>
#include <cmath>
#include <memory>
#include "ltimer.cpp"

using namespace std;


#define WIDTH 792
#define HEIGHT 792


int main(int argc, char **argv)
{
    torch::DeviceType device_type = at::kCUDA;
    const int MAX_FPS = 30;
    const int SCREEN_TICKS_PER_FRAME = 1000 / MAX_FPS;


    if (argc != 3)
    {
        std::cerr << "usage: example-app <path-to-exported-script-module> <other-module-path>\n";
        return -1;
    }

    torch::jit::script::Module module = torch::jit::load(argv[1], device_type);
    torch::jit::script::Module cppn = torch::jit::load(argv[2], device_type);

    std::cout << "ok1\n";
    // cppn.to(device_type);
    // module.to(device_type);
    std::cout << "ok2\n";

    // Create a vector of inputs for the xy meshgrid
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::tensor({(int) 200, (int) 200}).to(device_type));
    inputs.push_back(torch::ones({1}).to(device_type) * pow(3.0, 0.5));
    // std::vector<torch::jit::IValue> xy;
    // Calc the meshgrid
    at::Tensor xy = module.forward(inputs).toTensor().to(device_type);
    // at::Tensor xy = module.forward(inputs).toTensor();

    bool leftMouseButtonDown = false;
    double mouseX = 0.0;
    double mouseY = 0.0;
    bool quit = false;
    SDL_Event event;

    //The frames per second timer
    LTimer fpsTimer;

    //The frames per second cap timer
    LTimer capTimer;

    //In memory text stream
    std::stringstream timeText;

    //Start counting frames per second
    int countedFrames = 0;
    fpsTimer.start();

    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window *window = SDL_CreateWindow(
        "SDL2 Pixel Drawing", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WIDTH, HEIGHT, 0);

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, 0);
    SDL_Texture *texture = SDL_CreateTexture(renderer,
                           SDL_PIXELFORMAT_RGB888, SDL_TEXTUREACCESS_STATIC, WIDTH, HEIGHT);
    Uint8 *pixels = new Uint8[WIDTH * HEIGHT * 4];

    memset(pixels, 255, WIDTH * HEIGHT * 4 * sizeof(Uint8));

    // int max = 100;
    // for (int i = 0; i < max; ++i)
    while (!quit)
    {
        //Start cap timer
        capTimer.start();

        clock_t tStart = clock();
        SDL_UpdateTexture(texture, NULL, pixels, WIDTH * sizeof(Uint32));

        std::vector<torch::jit::IValue> inps;
        inps.push_back(xy);
        float extra[2] = {
            (float) ((mouseY / HEIGHT - 0.5) * 2.0),
            (float) ((mouseX / WIDTH  - 0.5) * 2.0)
        };
        inps.push_back(torch::tensor(extra).to(device_type));
        std::cout << extra[0] << " " << extra[1] << std::endl;

        at::Tensor t = cppn.forward(inps).toTensor();
        t = t * 255.0;
        t = at::reshape(t, -1);
        t = t.to(at::kCPU).toType(torch::kUInt8);
        // std::cout << "tensor dtype = " << t.dtype() << std::endl;
        auto array = t.accessor<unsigned char, 1>();
        for(int i = 0; i < array.size(0); i++)
        {
            pixels[i] = array[i];
        }

        //Handle events on queue
        while( SDL_PollEvent( &event ) != 0 )
        {
            if( event.type == SDL_QUIT )
            {
                quit = true;
            }
            switch (event.type)
            {
                case SDL_MOUSEBUTTONUP:
                    if (event.button.button == SDL_BUTTON_LEFT)
                        leftMouseButtonDown = false;
                    break;
                case SDL_MOUSEBUTTONDOWN:
                    if (event.button.button == SDL_BUTTON_LEFT)
                        leftMouseButtonDown = true;
                case SDL_MOUSEMOTION:
                    mouseX = event.motion.x;
                    mouseY = event.motion.y;
            }
        }

        //Calculate and correct fps
        countedFrames++;
        float avgFPS = countedFrames / ( fpsTimer.getTicks() / 1000.f );
        if( avgFPS > 2000000 )
        {
            avgFPS = 0;
        }
        //Set text to be rendered
        std::cout << "Average Frames Per Second (With Cap) " << avgFPS << std::endl;


        // SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
        printf("Time taken: %.6fs\n", (double)(clock() - tStart) / CLOCKS_PER_SEC);

        //If frame finished early
        int frameTicks = capTimer.getTicks();
        if( frameTicks < SCREEN_TICKS_PER_FRAME )
        {
            //Wait remaining time
            SDL_Delay( SCREEN_TICKS_PER_FRAME - frameTicks );
        }
    }

    delete[] pixels;
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);

    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
