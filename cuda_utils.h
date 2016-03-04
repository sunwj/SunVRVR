//
// Created by 孙万捷 on 16/3/4.
//

#ifndef SUNVRVR_CUDA_UTILS_H
#define SUNVRVR_CUDA_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK
#define CudaSafeCall(error) __cudaSafeCall(error, __FILE__, __LINE__)
inline void __cudaSafeCall(cudaError error, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
    if (error != cudaSuccess ) {
        printf("error: CudaSafeCall() failed at %s: %d with \" %s \"\n", file, line, cudaGetErrorString(error));
        exit( -1 );
    }
#endif
}

#endif //SUNVRVR_CUDA_UTILS_H
