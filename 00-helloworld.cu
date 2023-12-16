#include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("hello world from the GPU\n");
}

int main(void)
{
    hello_from_gpu<<<4, 4>>>();
    // 同步代码
    cudaDeviceSynchronize();
    return 0;
}