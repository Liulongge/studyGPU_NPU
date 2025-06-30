#include <cvcuda/OpConvertTo.hpp>
#include <nvcv/Image.hpp>
#include <nvcv/ImageBatch.hpp>
#include <nvcv/Tensor.hpp>
#include <cuda_runtime.h>

void NV12ToRGB(const void* nv12Data, void* rgbData, int width, int height) {
    // 创建 CUDA 流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 创建输入 NV12 张量
    nvcv::Tensor inTensor = nvcv::Tensor::Create(
        nvcv::TensorShape{1, height * 3 / 2, width, 1},
        nvcv::DataType{NVCV_DATA_TYPE_U8},
        nvcv::TensorLayout{NVCV_TENSOR_NHWC});
    
    // 创建输出 RGB 张量
    nvcv::Tensor outTensor = nvcv::Tensor::Create(
        nvcv::TensorShape{1, height, width, 3},
        nvcv::DataType{NVCV_DATA_TYPE_U8},
        nvcv::TensorLayout{NVCV_TENSOR_NHWC});

    // 将输入数据拷贝到张量
    cudaMemcpy(inTensor.exportData().basePtr(), nv12Data, 
               height * width * 3 / 2, cudaMemcpyHostToDevice);

    // 执行转换
    cvcuda::ConvertTo convertOp;
    convertOp(stream, inTensor, outTensor, NVCV_COLOR_CONVERSION_YUV2RGB_NV12);

    // 将结果拷贝回主机
    cudaMemcpy(rgbData, outTensor.exportData().basePtr(),
               height * width * 3, cudaMemcpyDeviceToHost);

    // 同步流并销毁
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
}


