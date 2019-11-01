#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <thread>
#include <SDL2/SDL.h>
#include <SDL2/SDL_render.h>
#include <cmath>
#include <memory>
#include "ltimer.cpp"
#include <signal.h>
#include "rtmidi/RtMidi.h"

using namespace std;

bool VERBOSE = false;


int main(int argc, char **argv)
{
    if (argc != 8)
    {
        std::cerr << "usage: example-app <path-to-exported-script-module> <cppn-module-path> <heatmap-module-path> <midi-port> <verbose> <inpWidth> <inpHeight>\n";
        return -1;
    }
    bool VERBOSE = (bool) argv[5];
    int inpWidth = (int) atoi(argv[6]);
    int inpHeight = (int) atoi(argv[7]);

    // disable pytorch gradients
    torch::NoGradGuard guard;
    torch::autograd::GradMode::set_enabled(false);

    double updateSpeedEMA = 0.75;
    // Map from midi byte identifying which knob was turned to position in vector
    std::map<int, int> knobChannelMappings = {
        {  3, 0 },
        {  9, 1 },
        { 12, 2 },
    };
    std::map<int, int> padChannelMappings = {
        { 36,  0 },
        { 37,  1 },
        { 38,  2 },
        { 39,  3 },
        { 40,  4 },
        { 41,  5 },
        { 42,  6 },
        { 43,  7 },
        { 44,  8 },
        { 45,  9 },
        { 46, 10 },
        { 47, 11 },
        { 48, 12 },
        { 49, 13 },
        { 50, 14 },
        { 51, 15 },
    };
    int numKnobs = padChannelMappings.size();
    // vector of zeros for each knob and its EMA
    std::vector<float> knobValues(numKnobs, 0.0);
    std::vector<float> knobValuesEMA(numKnobs, 0.0);

    torch::DeviceType device_type = at::kCUDA;
    const int MAX_FPS = 60;
    const int SCREEN_TICKS_PER_FRAME = 1000 / MAX_FPS;

    torch::jit::script::Module meshgridModule = torch::jit::load(argv[1], device_type);
    torch::jit::script::Module cppnModule = torch::jit::load(argv[2], device_type);
    torch::jit::script::Module heatmapModule = torch::jit::load(argv[3], at::kCPU);

    std::cout << "Loaded torch modules succesfully.\n";

    // Create a vector of inputs for the xy meshgrid
    std::vector<torch::jit::IValue> inputsToMeshgrid;
    int inputSize[2] = { inpWidth, inpHeight };
    at::Tensor inputSizeTensor = torch::tensor(inputSize).to(device_type);
    inputsToMeshgrid.push_back(inputSizeTensor);
    inputsToMeshgrid.push_back(torch::ones({1}).to(device_type) * pow(3.0, 0.5));
    std::vector<torch::jit::IValue> inputsToCPPN;
    // Calc the meshgrid
    at::Tensor xy = meshgridModule.forward(inputsToMeshgrid).toTensor().to(device_type);
    float extra[2] = {
        (float) ((knobValuesEMA[0] / 127.0 - 0.5) * 2.0),
        (float) ((knobValuesEMA[1] / 127.0 - 0.5) * 2.0)
    };
    inputsToCPPN.push_back(xy);
    inputsToCPPN.push_back(torch::tensor(extra).to(device_type));
    at::Tensor outputTensor = cppnModule.forward(inputsToCPPN).toTensor();
    // Calc the heatmap
    std::vector<torch::jit::IValue> inputsToHeatmap;
    std::vector<float> heatmapVals(16, 0.0);
    inputsToHeatmap.push_back(torch::tensor(heatmapVals));
    inputsToHeatmap.push_back(inputSizeTensor);
    inputsToHeatmap[0] = torch::tensor(heatmapVals);
    at::Tensor heatmapT = heatmapModule.forward(inputsToHeatmap).toTensor();
    std::cout << "HEATMAP size: " << heatmapT.size(2) << "x" << heatmapT.size(3) << std::endl;

    // Infer width and height from torch model output size
    int WIDTH = (int) outputTensor.size(2);
    int HEIGHT = (int) outputTensor.size(1);

    std::cout << "Window size: " << WIDTH << "x" << HEIGHT << std::endl;

    // Mouse stuff
    bool leftMouseButtonDown = false;
    double mouseX = 0.0;
    double mouseY = 0.0;
    bool quit = false;
    SDL_Event event;

    //The frames per second timer
    LTimer fpsTimer;

    //The frames per second cap timer
    LTimer capTimer;

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


    // Midi stuff
    unsigned int port = (unsigned int) atoi( argv[4] );
    RtMidiIn *midiin = 0;
    std::vector<unsigned char> message;
    int nBytes, i;
    double stamp;

    // RtMidiIn constructor
    try {
        midiin = new RtMidiIn();
    }
    catch ( RtMidiError &error ) {
        error.printMessage();
        exit( EXIT_FAILURE );
    }

    unsigned int nPorts = midiin->getPortCount();
    if ( port >= nPorts ) {
        delete midiin;
        std::cout << "port = " << port << " was an invalid port specifier!\n";
    }

    try {
        midiin->openPort( port );
    }
    catch ( RtMidiError &error ) {
        error.printMessage();
        goto cleanup;
    }

    memset(pixels, 255, WIDTH * HEIGHT * 4 * sizeof(Uint8));

    while (!quit)
    {
        // midi stuff
        while ((midiin->getMessage( &message )) != 0.0) {
            nBytes = message.size();
            int idx = (int) message[1];
            if (nBytes > 2) {
                // for ( i=0; i<nBytes; i++ )
                //     std::cout << "Byte " << i << " = " << (int)message[i] << ", ";
                // if ( nBytes > 0 )
                //     std::cout << "stamp = " << stamp << std::endl;
                if ( knobChannelMappings.find(idx) == knobChannelMappings.end() ) {
                    if ( padChannelMappings.find(idx) == padChannelMappings.end() ) {
                        // not found
                    } else {
                        heatmapVals[padChannelMappings[idx]] = (message[2] > 0.0) ? (float) message[2] : heatmapVals[padChannelMappings[idx]];
                    }
                } else {
                    knobValues[knobChannelMappings[idx]] = message[2];
                    // std::cout << "set position " << knobChannelMappings[idx] << " found via " << idx << " = "
                    //    << knobValues[knobChannelMappings[idx]] << std::endl;
                }
            }
        }

        //Decay heatmap
        for (int i = 0; i < heatmapVals.size(); ++i)
        {
            heatmapVals[i] = heatmapVals[i] * 0.8;
        }
        
        //Start cap timer
        capTimer.start();

        SDL_UpdateTexture(texture, NULL, pixels, WIDTH * sizeof(Uint32));

        // Update knob EMAs
        for (int i = 0; i < numKnobs; ++i)
        {
            knobValuesEMA[i] = knobValues[i] * updateSpeedEMA + knobValuesEMA[i] * (1.0 - updateSpeedEMA);
        }

        extra[0] = (float) ((knobValuesEMA[0] / 127.0 - 0.5) * 2.0);
        extra[1] = (float) ((knobValuesEMA[1] / 127.0 - 0.5) * 2.0);
        inputsToHeatmap[0] = torch::tensor(heatmapVals);
        heatmapT = heatmapModule.forward(inputsToHeatmap).toTensor().to(device_type);
        inputsToCPPN[0] = xy * (heatmapT * 0.5 * knobValues[2] / 127.0 + 1.0);
        inputsToCPPN[1] = torch::tensor(extra).to(device_type);

        if (VERBOSE)
            std::cout << extra[0] << " " << extra[1] << std::endl;

        outputTensor = cppnModule.forward(inputsToCPPN).toTensor();
        outputTensor = outputTensor * 255.0;
        outputTensor = at::reshape(outputTensor, -1);
        outputTensor = outputTensor.to(at::kCPU).toType(torch::kUInt8);
        // std::cout << "tensor dtype = " << outputTensor.dtype() << std::endl;
        auto array = outputTensor.accessor<unsigned char, 1>();
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
        if (VERBOSE) {
            std::cout << "Average Frames Per Second (With Cap) " << avgFPS << std::endl;
        }

        // SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
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
    delete midiin;

    return 0;

    // Clean up
 cleanup:
    delete midiin;

    return 0;
}
