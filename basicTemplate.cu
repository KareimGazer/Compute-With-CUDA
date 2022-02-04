#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define CHECK_FOR_ERRORS(err) if (err != cudaSuccess) {\
                                printf("%s:\n%s in %s in line %\n",cudaGetErrorName(err), cudaGetErrorString(err), __FILE__, __LINE__);\
                                exit(EXIT_FAILURE);\
                              }

__global__ void addKernel(float *c, const float*a, const float*b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int sizeBytes = arraySize * sizeof(float);
    const float a[arraySize] = { 1, 2, 3, 4, 5 };
    const float b[arraySize] = { 10, 20, 30, 40, 50 };
    float c[arraySize] = { 0 };

    float * dev_a = 0;
    float * dev_b = 0;
    float * dev_c = 0;

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    CHECK_FOR_ERRORS(cudaStatus);

    // allocating memory
    cudaStatus = cudaMalloc((void**)&dev_a, sizeBytes);
    CHECK_FOR_ERRORS(cudaStatus);

    cudaStatus = cudaMalloc((void**)&dev_b, sizeBytes);
    CHECK_FOR_ERRORS(cudaStatus);

    cudaStatus = cudaMalloc((void**)&dev_c, sizeBytes);
    CHECK_FOR_ERRORS(cudaStatus);

    // init with 0
    
    cudaStatus = cudaMemset((void*)dev_a, 0, sizeBytes);
    CHECK_FOR_ERRORS(cudaStatus);

    cudaStatus = cudaMemset((void*)dev_b, 0, sizeBytes);
    CHECK_FOR_ERRORS(cudaStatus);

    cudaStatus = cudaMemset((void*)dev_c, 0, sizeBytes);
    CHECK_FOR_ERRORS(cudaStatus);
    

    //copy to memory
    cudaStatus = cudaMemcpy(dev_a, a, sizeBytes, cudaMemcpyHostToDevice);
    CHECK_FOR_ERRORS(cudaStatus);

    cudaStatus = cudaMemcpy(dev_b, b, sizeBytes, cudaMemcpyHostToDevice);
    CHECK_FOR_ERRORS(cudaStatus);

    // Launch a kernel on the GPU with one thread for each element.
    addKernel << <1, arraySize >> > (dev_c, dev_a, dev_b);
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    CHECK_FOR_ERRORS(cudaStatus);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    CHECK_FOR_ERRORS(cudaStatus);
    
    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, sizeBytes, cudaMemcpyDeviceToHost);
    CHECK_FOR_ERRORS(cudaStatus);

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    CHECK_FOR_ERRORS(cudaStatus);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}
