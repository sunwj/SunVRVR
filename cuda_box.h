//
// Created by 孙万捷 on 16/3/3.
//

#ifndef SUNVRVR_CUDA_BOX_H
#define SUNVRVR_CUDA_BOX_H

#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "helper_math.h"
#include "cuda_ray.h"

class cudaBox
{
public:
    __host__ __device__ cudaBox(float width, float height, float depth)
    {
        float tmp = fmaxf(width, fmaxf(height, depth));
        width /= tmp;
        height /= tmp;
        depth /= tmp;

        top = make_float3(width, height, depth) * 0.5f;
        bottom = make_float3(width, height,depth) * -0.5f;
        invSize = 1.f / (top - bottom);

        //printf("box info:(%f, %f, %f) (%f, %f, %f)", bottom.x, bottom.y, bottom.z, top.x, top.y, top.z);
        //fflush(stdout);
    }

    __device__ bool Intersect(const cudaRay& ray, float* tNear, float* tFar) const
    {
        // compute intersection of ray with all six bbox planes
        float3 invR = make_float3(1.0f, 1.0f, 1.0f) / ray.dir;
        float3 tbot = invR * (bottom - ray.orig);
        float3 ttop = invR * (top - ray.orig);

        // re-order intersections to find smallest and largest on each axis
        float3 tmin = fminf(tbot, ttop);
        float3 tmax = fmaxf(tbot, ttop);

        // find the largest tmin and the smallest tmax
        float largest_tmin = fmaxf(tmin.x, fmaxf(tmin.y, tmin.z));
        float smallest_tmax = fminf(tmax.x, fminf(tmax.y, tmax.z));

        *tNear = largest_tmin;
        *tFar = smallest_tmax;

        return smallest_tmax > largest_tmin;
    }

    __device__ float3 GetTexCoord(const float3& pt)
    {
        return (pt - bottom) * invSize;
    }

    __device__ bool OutOfBound(const float3& pt)
    {
        if(pt.x > top.x || pt.y > top.y || pt.z > top.z)
            return true;
        if(pt.x < bottom.x || pt.y < bottom.y || pt.z < bottom.z)
            return true;
        return false;
    }

public:
    float3 top;
    float3 bottom;
    float3 invSize;
};

#endif //SUNVRVR_CUDA_BOX_H
