// -*- compile-command: "nvcc arch sm_50 -Xptxas=-v -cubin kernel.cu"; -*-

//
//
//

#ifdef __cplusplus
extern "C" {
#endif

#include "assert_cuda.h"

#ifdef __cplusplus
}
#endif

//
//
//

#define PXL_KERNEL_THREADS_PER_BLOCK  256 // enough for 4Kx2 monitor

//
//
//

surface<void,cudaSurfaceType2D> surf;

//
//
//

union pxl_rgbx_24
{
  uint1       b32;

  struct {
    unsigned  r  : 8;
    unsigned  g  : 8;
    unsigned  b  : 8;
    unsigned  na : 8;
  };
};

//
//
//

extern "C"
__global__
void
pxl_kernel(const int width, const int height)
{
  // pixel coordinates
  const int idx = (blockDim.x * blockIdx.x) + threadIdx.x;
  const int x   = idx % width;
  const int y   = idx / width;

#if 1

  // pixel color
  const int          t    = (unsigned int)clock() / 1100000; // 1.1 GHz
  const int          xt   = (idx + t) % width;
  const unsigned int ramp = (unsigned int)(((float)xt / (float)(width-1)) * 255.0f + 0.5f);
  const unsigned int bar  = ((y + t) / 32) & 3;

  union pxl_rgbx_24  rgbx;

  rgbx.r  = (bar == 0) || (bar == 1) ? ramp : 0;
  rgbx.g  = (bar == 0) || (bar == 2) ? ramp : 0;
  rgbx.b  = (bar == 0) || (bar == 3) ? ramp : 0;
  rgbx.na = 255;

#else // DRAW A RED BORDER TO VALIDATE FLIPPED BLIT

  const bool        border = (x == 0) || (x == width-1) || (y == 0) || (y == height-1);
  union pxl_rgbx_24 rgbx   = { border ? 0xFF0000FF : 0xFF000000 };
  
#endif

  surf2Dwrite(rgbx.b32, // even simpler: (unsigned int)clock()
    surf,
    x*sizeof(rgbx),
    y,
    cudaBoundaryModeZero); // squelches out-of-bound writes
}

//
//
//

extern "C"
cudaError_t
pxl_kernel_launcher(cudaArray_const_t array,
                    const int         width,
                    const int         height,
                    cudaEvent_t       event,
                    cudaStream_t      stream)
{
  cudaError_t cuda_err;

  // cuda_err = cudaEventRecord(event,stream);

  cuda_err = cuda(BindSurfaceToArray(surf,array));

  if (cuda_err)
    return cuda_err;

  const int blocks = (width * height + PXL_KERNEL_THREADS_PER_BLOCK - 1) / PXL_KERNEL_THREADS_PER_BLOCK;

  // cuda_err = cudaEventRecord(event,stream);

  if (blocks > 0)
    pxl_kernel<<<blocks,PXL_KERNEL_THREADS_PER_BLOCK,0,stream>>>(width,height);

  // cuda_err = cudaStreamWaitEvent(stream,event,0);
  
  return cudaSuccess;
}

//
//
//
