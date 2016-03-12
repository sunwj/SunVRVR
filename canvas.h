//
// Created by 孙万捷 on 16/3/4.
//

#ifndef SUNVRVR_CANVAS_H
#define SUNVRVR_CANVAS_H

#include <QGLWidget>

#include <cuda_gl_interop.h>

#include "cuda_utils.h"
#include "pathtracer.h"

class Canvas : public QGLWidget
{
    Q_OBJECT
public:
    explicit Canvas(const QGLFormat& format, QWidget* parent = 0);
    virtual ~Canvas();

protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();

private:
    GLuint leftPBO = 0;
    GLuint rightPBO = 0;
    cudaGraphicsResource* leftResource;
    cudaGraphicsResource* rightResource;
    uchar4* leftImg;
    uchar4* rightImg;

    cudaBox volumeBox;
    cudaCamera camera;
};


#endif //SUNVRVR_CANVAS_H
