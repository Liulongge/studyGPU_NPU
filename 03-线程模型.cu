
#include <stdio.h>

// 推广到多维
// 1、cuda可以组织三维网格和线程块
// 2、blockIdx和threadIdx是类型为uint32的变量，该类型是一个结构体，
// 具有x,y,z三个成员(3个成员都是无符号类型的成员)：
// blockIdx.x,  blockIdx.y,  blockIdx.z
// threadIdx.x, threadIdx.y, threadIdx.z
// 3、gridDim和blockDim是类型为dim3的变量，该类型是一个结构体，具有xyz三个成员：
// gridDim.x,  graiDim.y,  gridDim.z
// blockDim.x, blockDim.y, blockDim.z
// 4、取值范围：
// blockIdx.x: [0, gridDim.x - 1]
// blockIdx.y: [0, gridDim.y - 1]
// blockIdx.z: [0, gridDim.z - 1]

// threadIdx.x: [0, blockDim.x - 1]
// threadIdx.x: [0, blockDim.y - 1]
// threadIdx.x: [0, blockDim.z - 1]

// 注意内建变量只有在核函数有效，且无需定义 

// 定义多维网络和线程块(c++构造函数语法)
// dim3 grid_size(Gx, Gy, Gz)
// dim3 block_size(Bx, By, Bz)
// 举例：定义一个2x2x1的网络，5x3x1的线程块
// dim grid_size(2, 2) // 等价于 dim3 grid_size(2, 2, 1)
// dim block_size(5, 3) // 等价于 dim3 block_size(5, 3, 1)

// 多维线程块中线程索引
// int tid = threadIdx.z * blockDim.x * blockDim.y +
//           threadIdx.y * blockDim.x + threadIdx.x
// 多维网络中的线程块索引
// int bid = blockIdx.z * gridDim.x * gridDim.y +
//           blockIdx.y * gridDim.x + blockIdx.x

// 网格大小限制：
// gridDim.x最大值 2^31 - 1
// gridDim.y最大值 2^16 - 1
// gridDim.y最大值 2^16 - 1
// 线程大小限制：
// blockDim.x最大值 1024
// blockDim.y最大值 1024
// blockDim.z最大值 64


__global__ void hello_from_gpu()
{
    // 获取线程块索引值
    const int bid = blockIdx.x;
    // 获取线程索引值
    const int tid = threadIdx.x;
    // 获取线程块个数
    const int dim = blockDim.x;
    // 整个网格中线程索引
    const int gid = tid + bid * dim;

    printf("Hello World from block %d and thread %d, global id %d\n", bid, tid, gid);
}

int main(void)
{
    // <<<n, m>>>括号中参数用来指明核函数执行线程
    // n: 线程块个数
    // m: 每个线程块线程数量
    hello_from_gpu<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}