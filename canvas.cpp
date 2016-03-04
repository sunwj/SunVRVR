//
// Created by 孙万捷 on 16/3/4.
//

#include "canvas.h"

Canvas::Canvas(const QGLFormat &format, QWidget *parent) : QGLWidget(format, parent)
{

}

Canvas::~Canvas()
{

}

void Canvas::initializeGL()
{
    makeCurrent();

    glClearColor(0.f, 0.f, 0.f, 0.f);
    glClear(GL_COLOR_BUFFER_BIT);

    glGenBuffers(1, &leftPBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, leftPBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 640 * 640 * 4, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glGenBuffers(1, &rightPBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, rightPBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 640 * 640 * 4, NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    CudaSafeCall(cudaGraphicsGLRegisterBuffer(&leftResource, leftPBO, cudaGraphicsMapFlagsNone));
    CudaSafeCall(cudaGraphicsGLRegisterBuffer(&rightResource, rightPBO, cudaGraphicsMapFlagsNone));
}

void Canvas::resizeGL(int w, int h)
{
    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-640, 640, -640, 640, 0, 10000000);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void Canvas::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT);
    /*glBegin(GL_TRIANGLES);
    glColor3f(1.f, 1.f, 1.f);
    glVertex2d(0.f, 0.5f);
    glColor3f(1.f, 1.f, 1.f);
    glVertex2d(0.5f, -0.5f);
    glColor3f(1.f, 1.f, 1.f);
    glVertex2d(-0.5f, -0.5f);
    glEnd();*/

    size_t size;
    CudaSafeCall(cudaGraphicsMapResources(1, &leftResource, 0));
    CudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&leftImg, &size, leftResource));
    CudaSafeCall(cudaGraphicsMapResources(1, &rightResource, 0));
    CudaSafeCall(cudaGraphicsResourceGetMappedPointer((void**)&rightImg, &size, rightResource));

    render3d(leftImg, rightImg, 0);
    CudaSafeCall(cudaDeviceSynchronize());

    CudaSafeCall(cudaGraphicsUnmapResources(1, &leftResource, 0));
    CudaSafeCall(cudaGraphicsUnmapResources(1, &rightResource, 0));

    glRasterPos2i(-640, -640);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, leftPBO);
    glDrawPixels(640, 640, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    glRasterPos2i(0, -640);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, rightPBO);
    glDrawPixels(640, 640, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}
