# 参考
https://github.com/sangyc10/CUDA-code

        GPU: 数据运算
        CPU: 逻辑运算

        GPU性能指标: 核心数、GPU显存容量、GPU计算峰值、显存带宽

# cpu + gpu异构架构
        gpu不能单独计算，cpu+gpu组成异构计算架构
        cpu起到控制作用，一般称为主机(host);
        gpu可以看作cpu的协处理器，一般称为设备(device)
        主机和设备之前内存访问一般通过PCIe总线链接
![cpu+gpu异构架构.jpg](./cpu+gpu异构架构.jpg)

# cuda编程语言
        2006年，英伟达发布cuda
        cuda旨在支持各种语言和应用程序编程接口
        最初基于c语言，目前越来越多支持c++
        cuda还支持pyton编写
![cuda编程语言](./cuda编程语言.jpg)

# cuda运行时api
        cuda提供两层api接口，cuda驱动(driver)api和cuda运行时(runtime)api
        两种api调用性能几乎无差异，课程使用操作对用户更加友好的runtime api
![cuda编程开发环境](./cuda编程开发环境.jpg)


# 相关命令
## NVCC -V是什么？
        NVCC -V是NVIDIA Cuda Compiler的版本检查命令，主要用于显示当前系统上安装的CUDA版本和编译器版本。该命令可以帮助CUDA开发者确认所使用的软件是否是最新版本，以保证程序的性能和稳定性。
        显示一下信息:
        1、NVIDIA Cuda编译器驱动程序的名称：nvcc
        2、NVIDIA Corporation的版权信息
        3、编译器的编译时间：Built on...
        4、CUDA编译工具的版本号：Cuda compilation tools, release...
        5、CUDA编译工具编译时使用的版本号：Build cuda_...
## nvidia-smi
        nvidia-smi（NVIDIA System Management Interface）是一种命令行实用程序，用于监控和管理 NVIDIA GPU（图形处理器）的状态和性能。它提供了一种简单而强大的方式来获取有关 GPU 的实时信息，并且可以用于诊断、优化和管理 GPU 资源。
        信息如下:
                1、GPU：本机中的GPU编号，从0开始，上图为0，1，2，3四块GPU
                2、Fan：风扇转速（0%-100%），N/A表示没有风扇
                3、Name：GPU名字/类型，上图四块均为NVIDIA GeForce RTX 3080
                4、Temp：GPU温度（GPU温度过高会导致GPU频率下降）
                5、Perf：性能状态，从P0（最大性能）到P12（最小性能），上图均为P2
                6、Pwr：Usager/Cap：GPU功耗，Usage表示用了多少，Cap表示总共多少
                7、Persistence-M：持续模式状态，持续模式耗能大，但在新的GPU应用启动时花费时间更少，上图均为On
                8、Bus-Id：GPU总线
                9、Disp.A：Display Active，表示GPU是否初始化
                10、Memory-Usage：显存使用率
                11、Volatile GPU-UTil：GPU使用率，与显存使用率的区别可参考显存与GPU
                12、Uncorr. ECC：是否开启错误检查和纠错技术，0/DISABLED，1/ENABLED，上图均为N/A
                13、Compute M：计算模式，0/DEFAULT，1/EXCLUSIVE_PROCESS，2/PROHIBITED，上图均为Default
                14、Processes：显示每个进程占用的显存使用率、进程号、占用的哪个GPU
        其他命令：
                1、查看gpu详细信息：nvidia-smi -q
                2、查询特定gpu详细信息：nvidia-smi -q -i 0
                3、显示gpu特定信息：nvidia-smi -q -i 0 -d MEMORY
                4、帮助命令：nvidia-smi -h
![nvidia-smi](./nvidia-smi.jpg)


# nvcc编译命令
        nvcc test.cu -o demo
# 核函数
        1、核函数在GPU上进行并行执行, 主机对设备的调用通过核函数
        2、注意：
                2.1、限定词__global__; 
                2.2、返回值必须为void
        3、形式：
                3.1、
                __global__ void kernel_function(argument arg)
                {
                        printf("hello world from the GPU\n")
                }
                3.2、
                void __global__ kernel_function(argument arg)
                {
                        printf("hello world from the GPU\n")
                }
        注意事项：
                1、核函数只能访问GPU内存
                2、核函数不能使用变长参数
                3、核函数不能使用全局变量
                4、核函数不能使用函数指针
                5、核函数具有异步性
## cuda程序编写流程
        int main(void)
        {
                主机代码
                核函数调用
                主机代码
                return 0;
        }

# cuda线程模型
## 线程模型结构
        一、线程模型重要概念
                1、grid 网络
                2、block 线程块
                一个Kernel函数对应一个Grid。
                每个Grid中的任务是一定的。当要执行这些任务的时候，每一个Grid又把任务分成一部分一部分的Block（线程块），Block中间有若干Thread（线程），再分成线程来完成。
        二、线程分块是逻辑上的划分，物理上先层不分块
        三、配置线程：<<<grid_size, block_size>>>
                grid_size: 几个block
                block_size: 几个thread
        四、最大允许线程块大小：1024，最大允许网格大小2^31 - 1(针对一维网络) 
![线程模型结构](./线程模型结构.jpg)

## 线程组织管理
## 网格和线程快限制