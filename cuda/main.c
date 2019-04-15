//
//
//

#include <glad/glad.h>
#include <GLFW/glfw3.h>

//
//
//

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

//
//
//

#include <cuda_gl_interop.h>

//
//
//

#include "assert_cuda.h"
#include "interop.h"

//
// FPS COUNTER FROM HERE:
//
// http://antongerdelan.net/opengl/glcontext2.html
//

static
void
pxl_glfw_fps(GLFWwindow* window)
{
  // static fps counters
  static double stamp_prev  = 0.0;
  static int    frame_count = 0;

  // locals
  const double stamp_curr = glfwGetTime();
  const double elapsed    = stamp_curr - stamp_prev;
  
  if (elapsed > 0.5)
    {
      stamp_prev = stamp_curr;
      
      const double fps = (double)frame_count / elapsed;

      int  width, height;
      char tmp[64];

      glfwGetFramebufferSize(window,&width,&height);
  
      sprintf_s(tmp,64,"(%u x %u) - FPS: %.2f",width,height,fps);

      glfwSetWindowTitle(window,tmp);

      frame_count = 0;
    }

  frame_count++;
}

//
//
//

static
void
pxl_glfw_error_callback(int error, const char* description)
{
  fputs(description,stderr);
}

static
void
pxl_glfw_key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    glfwSetWindowShouldClose(window, GL_TRUE);
}

static
void
pxl_glfw_init(GLFWwindow** window, const int width, const int height)
{
  //
  // INITIALIZE GLFW/GLAD
  //
  
  glfwSetErrorCallback(pxl_glfw_error_callback);

  if (!glfwInit())
    exit(EXIT_FAILURE);
 
  glfwWindowHint(GLFW_DEPTH_BITS,            0);
  glfwWindowHint(GLFW_STENCIL_BITS,          0);

  glfwWindowHint(GLFW_SRGB_CAPABLE,          GL_TRUE);

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);

  glfwWindowHint(GLFW_OPENGL_PROFILE,        GLFW_OPENGL_CORE_PROFILE);

#ifdef PXL_FULLSCREEN
  GLFWmonitor*       monitor = glfwGetPrimaryMonitor();
  const GLFWvidmode* mode    = glfwGetVideoMode(monitor);
  *window                    = glfwCreateWindow(mode->width,mode->height,"GLFW / CUDA Interop",monitor,NULL);
#else
  *window = glfwCreateWindow(width,height,"GLFW / CUDA Interop",NULL,NULL);
#endif

  if (*window == NULL)
    {
      glfwTerminate();
      exit(EXIT_FAILURE);
    }

  glfwMakeContextCurrent(*window);

  // set up GLAD
  gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

  // ignore vsync for now
  glfwSwapInterval(0);

  // only copy r/g/b
  glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_FALSE);

  // enable SRGB 
  // glEnable(GL_FRAMEBUFFER_SRGB);
}

//
//
//

static
void
pxl_glfw_window_size_callback(GLFWwindow* window, int width, int height)
{
  // get context
  struct pxl_interop* const interop = glfwGetWindowUserPointer(window);

  pxl_interop_size_set(interop,width,height);
}

//
//
//

cudaError_t
pxl_kernel_launcher(cudaArray_const_t array, 
                    const int         width, 
                    const int         height,
                    cudaEvent_t       event,
                    cudaStream_t      stream);

//
//
//

int
main(int argc, char* argv[])
{
  //
  // INIT GLFW
  //
  GLFWwindow* window;

  pxl_glfw_init(&window,1024,1024);

  //
  // INIT CUDA
  //
  cudaError_t cuda_err;
  
  int gl_device_id,gl_device_count;
  cuda_err = cuda(GLGetDevices(&gl_device_count,&gl_device_id,1,cudaGLDeviceListAll));

  int cuda_device_id = (argc > 1) ? atoi(argv[1]) : gl_device_id;
  cuda_err = cuda(SetDevice(cuda_device_id));

  //
  // MULTI-GPU?
  //
  const bool multi_gpu = gl_device_id != cuda_device_id;

  //
  // INFO
  //
  struct cudaDeviceProp props;

  cuda_err = cuda(GetDeviceProperties(&props,gl_device_id));
  printf("GL   : %-24s (%2d)\n",props.name,props.multiProcessorCount);

  cuda_err = cuda(GetDeviceProperties(&props,cuda_device_id));
  printf("CUDA : %-24s (%2d)\n",props.name,props.multiProcessorCount);

  //
  // CREATE CUDA STREAM & EVENT
  //
  cudaStream_t stream;
  cudaEvent_t  event;

  cuda_err = cuda(StreamCreateWithFlags(&stream,cudaStreamDefault));   // optionally ignore default stream behavior
  cuda_err = cuda(EventCreateWithFlags(&event,cudaEventBlockingSync)); // | cudaEventDisableTiming);

  //
  // CREATE INTEROP
  //
  // TESTING -- DO NOT SET TO FALSE, ONLY TRUE IS RELIABLE
  struct pxl_interop* const interop = pxl_interop_create(true /*multi_gpu*/,2);

  //
  // RESIZE INTEROP
  //
  
  int width, height;

  // get initial width/height
  glfwGetFramebufferSize(window,&width,&height);

  // resize with initial window dimensions
  cuda_err = pxl_interop_size_set(interop,width,height);

  //
  // SET USER POINTER AND CALLBACKS
  //
  glfwSetWindowUserPointer      (window,interop);
  glfwSetKeyCallback            (window,pxl_glfw_key_callback);
  glfwSetFramebufferSizeCallback(window,pxl_glfw_window_size_callback);
  
  //
  // LOOP UNTIL DONE
  //
  while (!glfwWindowShouldClose(window))
    {
      //
      // MONITOR FPS
      //
      pxl_glfw_fps(window);
      
      //
      // EXECUTE CUDA KERNEL ON RENDER BUFFER
      //
      int         width,height;
      cudaArray_t cuda_array;

      pxl_interop_size_get(interop,&width,&height);

      cuda_err = pxl_interop_map(interop,stream);

      cuda_err = pxl_kernel_launcher(pxl_interop_array_get(interop),
                                     width,
                                     height,
                                     event,
                                     stream);

      cuda_err = pxl_interop_unmap(interop,stream);

      //
      // BLIT & SWAP FBO
      // 
      pxl_interop_blit(interop);
      // pxl_interop_clear(interop);
      pxl_interop_swap(interop);

      //
      // SWAP WINDOW
      //
      glfwSwapBuffers(window);

      //
      // PUMP/POLL/WAIT
      //
      glfwPollEvents(); // glfwWaitEvents();
    }

  //
  // CLEANUP
  //
  pxl_interop_destroy(interop);
  
  glfwDestroyWindow(window);

  glfwTerminate();

  cuda(DeviceReset());

  // missing some clean up here

  exit(EXIT_SUCCESS);
}

//
//
//
