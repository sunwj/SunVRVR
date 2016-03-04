//
// Created by 孙万捷 on 16/3/4.
//

#ifndef SUNVRVR_PATHTRACER_H
#define SUNVRVR_PATHTRACER_H

#include "cuda_camera.h"
#include "cuda_box.h"

extern "C" void render3d(uchar4* leftImg, uchar4* rightImg, unsigned int frameNo);

#endif //SUNVRVR_PATHTRACER_H
