// 1、核函数在GPU上进行并行执行, 主机对设备的调用通过核函数
// 2、注意：
//      2.1、限定词__global__; 
//      2.2、返回值必须为void
// 3、形式：
//      3.1、__global__ void kernel_function(argument arg)
//      {
//          printf("hello world from the GPU\n")
//      }
//      3.2、void __global__ kernel_function(argument arg)
//      {
//          printf("hello world from the GPU\n")
//      }
// 注意事项：
// 1、核函数只能访问GPU内存
// 2、核函数不能使用变长参数
// 3、核函数不能使用全局变量
// 4、核函数不能使用函数指针
// 5、核函数具有异步性
// cuda程序编写流程
// int main(void)
// {
//      主机代码
//      核函数调用
//      主机代码
//      return 0;
// }

#include <stdio.h>
// __global__限定符，这是一个核函数
// 核函数不支持c++的iostream
__global__ void hello_from_gpu()
{
    printf("Hello World from GPU\n");
}

int main(void)
{
    // <<<n, m>>>括号中参数用来指明核函数执行线程
    // n: 线程块个数
    // m: 每个线程块线程数量
    hello_from_gpu<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}