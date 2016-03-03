#include "transferfunction.h"

TransferFunction::TransferFunction(vtkSmartPointer<vtkPiecewiseFunction> otf, vtkSmartPointer<vtkColorTransferFunction> ctf, QObject *parent) : QObject(parent)
{
    opacityTF = otf;
    colorTF = ctf;

    this->otf = QSharedPointer<ctkTransferFunction>(new ctkVTKPiecewiseFunction(opacityTF));
    this->ctf = QSharedPointer<ctkTransferFunction>(new ctkVTKColorTransferFunction(colorTF));

    connect(this->otf.data(), SIGNAL(changed()), this, SLOT(onOpacityTFChanged()));
    connect(this->ctf.data(), SIGNAL(changed()), this, SLOT(onColorTFChanged()));

    compositeTex = 0;

    // initialize each table
    opacityTF->GetTable(0.0, 1.0, TABLE_SIZE, opacityTable);
    colorTF->GetTable(0.0, 1.0, TABLE_SIZE, colorTable);
    CompositeTable();

    channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    CudaSafeCall(cudaMallocArray(&array, &channelDesc, TABLE_SIZE));
    CudaSafeCall(cudaMemcpyToArray(array, 0, 0, compositeTable, sizeof(float) * TABLE_SIZE * 4, cudaMemcpyHostToDevice));

    memset(&resourceDesc, 0, sizeof(resourceDesc));
    resourceDesc.resType = cudaResourceTypeArray;
    resourceDesc.res.array.array = array;

    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.normalizedCoords = true;
    texDesc.readMode = cudaReadModeElementType;

    CudaSafeCall(cudaCreateTextureObject(&compositeTex, &resourceDesc, &texDesc, NULL));
}

TransferFunction::~TransferFunction()
{
    if(compositeTex)
        CudaSafeCall(cudaDestroyTextureObject(compositeTex));

    CudaSafeCall(cudaFreeArray(array));
}

void TransferFunction::SaveCurrentTFConfiguration()
{
    std::ofstream output("tf.txt");
    if(!output.good())
    {
        std::cerr<<"unable to open tf.txt"<<std::endl;
        exit(0);
    }
    for(unsigned int i = 0; i < TABLE_SIZE * 4; ++i)
    {
        output<<compositeTable[i]<<',';
    }
    output.close();
}

void TransferFunction::onOpacityTFChanged()
{
    //std::cout<<"Opacity changed"<<std::endl;
    if(compositeTex)
    {
        CudaSafeCall(cudaDestroyTextureObject(compositeTex));
        compositeTex = 0;
    }

    opacityTF->GetTable(0.0, 1.0, TABLE_SIZE, opacityTable);
    size_t j = 3;
    for(size_t i = 0; i < TABLE_SIZE; ++i)
    {
        compositeTable[j] = opacityTable[i];
        j += 4;
    }
    //CompositeTable();

    CudaSafeCall(cudaMemcpyToArray(array, 0, 0, compositeTable, sizeof(float) * TABLE_SIZE * 4, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaCreateTextureObject(&compositeTex, &resourceDesc, &texDesc, NULL));

    Changed();
}

void TransferFunction::onColorTFChanged()
{
    //std::cout<<"Color changed"<<std::endl;
    if(compositeTex)
    {
        CudaSafeCall(cudaDestroyTextureObject(compositeTex));
        compositeTex = 0;
    }

    colorTF->GetTable(0.0, 1.0, TABLE_SIZE, colorTable);
    size_t j = 0, k = 0;
    for(size_t i = 0; i < TABLE_SIZE; ++i)
    {
        compositeTable[j++] = colorTable[k++];
        compositeTable[j++] = colorTable[k++];
        compositeTable[j++] = colorTable[k++];
        j++;
    }
    //CompositeTable();

    CudaSafeCall(cudaMemcpyToArray(array, 0, 0, compositeTable, sizeof(float) * TABLE_SIZE * 4, cudaMemcpyHostToDevice));
    CudaSafeCall(cudaCreateTextureObject(&compositeTex, &resourceDesc, &texDesc, NULL));

    Changed();
}
