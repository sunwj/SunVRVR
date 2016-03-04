//
// Created by 孙万捷 on 16/3/4.
//

#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include "cuda_box.h"
#include "cuda_camera.h"

#define IMAGE_WIDTH 640

__host__ __device__ unsigned int wangHash(unsigned int a)
{
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);

    return a;
}

__global__ void render_left(uchar4* img, const cudaBox volumeBox, const cudaCamera camera, unsigned int hashedFrameNo)
{
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    auto idy = blockDim.y * blockIdx.y + threadIdx.y;
    auto offset = idy * 640 + idx;
    curandState rng;
    curand_init(hashedFrameNo + offset, 0, 0, &rng);

    cudaRay ray;
    camera.GenerateRay(idx, idy, &rng, &ray);

    float tNear, tFar;
    if(!volumeBox.Intersect(ray, &tNear, &tFar))
        img[offset] = make_uchar4(0, 0, 0, 0);
    else
        img[offset] = make_uchar4(255, 0, 0, 255);
}

__global__ void render_right(uchar4* img, const cudaBox volumeBox, const cudaCamera camera, unsigned int hashedFrameNo)
{
    auto idx = blockDim.x * blockIdx.x + threadIdx.x;
    auto idy = blockDim.y * blockIdx.y + threadIdx.y;
    auto offset = idy * 640 + idx;
    curandState rng;
    curand_init(hashedFrameNo + offset, 0, 0, &rng);

    cudaRay ray;
    camera.GenerateRay(idx, idy, &rng, &ray);

     float tNear, tFar;
     if(!volumeBox.Intersect(ray, &tNear, &tFar))
         img[offset] = make_uchar4(0, 0, 0, 0);
     else
        img[offset] = make_uchar4(0, 255, 0, 255);
}

extern "C" void render3d(uchar4* leftImg, uchar4* rightImg, unsigned int frameNo)
{
    dim3 blockSize(16, 16);
    dim3 gridSize(640 / blockSize.x, 640 / blockSize.y);

    cudaBox volumeBox(1.f, 1.f, 1.f);
    cudaCamera camera(make_float3(0.f, 0.f, 3.f), make_float3(1.f, 0.f, 0.f), make_float3(0.f, 1.f, 0.f), make_float3(0.f, 0.f, 1.f), 75.f, 640, 640);

    render_left<<<gridSize, blockSize>>>(leftImg, volumeBox, camera, frameNo);
    render_right<<<gridSize, blockSize>>>(rightImg, volumeBox, camera, frameNo);
}