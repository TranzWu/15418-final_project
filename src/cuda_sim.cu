#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>

#include "cuda_sim.h"

struct GlobalConstants {
    size_t numberOfParticles;

    float boxSize;

    float dudr;
    float r_cut;
    float u_cut;
    float delta;

    float* positions;
    float* velocities;
    float* forces;

    float* kinetic;
    float* potential;
};
__constant__ GlobalConstants params;

/* Device functions */

__device__ float calculateDistance(size_t i, size_t j)
{
    float boxSize = params.boxSize;

    float distance = 0;
    for (int k = 0; k < 3; k++){
        float diff = params.positions[i * 3 + k] - params.positions[j * 3 + k];
        if (diff < -boxSize/2) diff += boxSize;
        if (diff > boxSize/2) diff -= boxSize;
        distance += diff * diff;
    }

    distance = std::sqrt(distance);

    return distance;
}

/* Kernels */

__global__ void calculateForceKernel()
{
    int numberOfParticles = params.numberOfParticles;
    float boxSize = params.boxSize;
    float dudr = params.dudr;
    float r_cut = params.r_cut;
    float u_cut = params.u_cut;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index % numberOfParticles;
    int j = index / numberOfParticles;

    if (i >= numberOfParticles || j >= numberOfParticles) return;
    if (j <= i) return;
    if ((i % 3 == 0) && (j - i == 1) || (i % 3 == 0) && (j - i == 2) ) return;

    float dist = calculateDistance(i, j);
    if (dist >= r_cut) return;

    float u_actual = 4 * (1/std::pow(dist, 12) - 1/std::pow(dist, 6));
    atomicAdd(params.potential, u_actual - u_cut - (dist - r_cut) * dudr);

    for (int k = 0; k < 3; k++)
    {
        float r = params.positions[i * 3 + k] - params.positions[j * 3 + k];
        if (r < -boxSize/2) r += boxSize;
        if (r > boxSize/2) r -= boxSize;

        float f = r * (48/pow(dist, 14) - 24/pow(dist, 8) + dudr/dist);
        atomicAdd(&params.forces[i * 3 + k], f);
        atomicAdd(&params.forces[j * 3 + k], -1 * f);
    }

}

__global__ void calculateBondAngleKernel()
{
    int numberOfParticles = params.numberOfParticles;
    float boxSize = params.boxSize;
    double spring = 1000;
    double x0 = 0.8;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numberOfParticles * 2) return;
    int h = index % 2 + 1;
    int i = (index / 2) * 3;

    float dist = calculateDistance(i, i + h);

    for (int k = 0; k < 3; k++){
        double r = params.positions[i * 3 + k] - params.positions[(i+h) * 3 + k];
        if (r < -boxSize/2) r += boxSize;
        if (r > boxSize/2) r -= boxSize;
        atomicAdd(params.potential, 0.5 * spring * pow(dist - x0, 2));
        float f = -spring * (dist - x0) * r / dist;
        atomicAdd(&params.forces[i * 3 + k], f);
        atomicAdd(&params.forces[(i + h) * 3 + k], -f);
    }
}

__global__ void calculateKineticKernel()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > params.numberOfParticles * 3) return;
    int i = index / 3;
    int k = index % 3;

    atomicAdd(params.kinetic, 0.5 * pow(params.velocities[i * 3 + k], 2));
}

__global__ void updatePositionKernel()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > params.numberOfParticles * 3) return;
    int i = index / 3;
    int k = index % 3;

    params.positions[i * 3 + k] += params.velocities[i * 3 + k] * params.delta;
    if (params.positions[i * 3 + k] > params.boxSize){
        params.positions[i * 3 + k] -= params.boxSize;
    }
    if (params.positions[i * 3 + k] < 0){
        params.positions[i * 3 + k] += params.boxSize;
    }
}

__global__ void updateVelocityKernel()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > params.numberOfParticles * 3) return;
    int i = index / 3;
    int k = index % 3;

    float a = params.forces[i * 3 + k] * params.delta / 2;
    params.velocities[i * 3 + k] += a;
}

/* Interface */

void CudaSim::getPositions(float *out)
{
    cudaMemcpy(out, positions, 3 * numberOfParticles * sizeof(float), cudaMemcpyDeviceToHost);
}

void CudaSim::getVelocities(float *out)
{
    cudaMemcpy(out, velocities, 3 * numberOfParticles * sizeof(float), cudaMemcpyDeviceToHost);
}

float CudaSim::getKinetic()
{
    float out;
    cudaMemcpy(&out, kinetic, sizeof(float), cudaMemcpyDeviceToHost);
    return out;
}

float CudaSim::getPotential()
{
    float out;
    cudaMemcpy(&out, potential, sizeof(float), cudaMemcpyDeviceToHost);
    return out;
}

void printDeviceInfo()
{
    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}

CudaSim::CudaSim(size_t numberOfParticles_in, float boxSize_in, float dudr_in,
                 float r_cut_in, float u_cut_in, float delta_in, float* positions_init)
{
    printDeviceInfo();

    // Initialize instance variables
    numberOfParticles = numberOfParticles_in;
    boxSize = boxSize_in;
    dudr = dudr_in;
    r_cut = r_cut_in;
    u_cut = u_cut_in;
    delta = delta_in;
    cudaMalloc(&positions, sizeof(float) * 3 * numberOfParticles);
    cudaMemcpy(positions, positions_init, sizeof(float) * 3 * numberOfParticles, cudaMemcpyHostToDevice);
    cudaMalloc(&velocities, sizeof(float) * 3 * numberOfParticles);
    cudaMemset(velocities, 0, sizeof(float) * 3 * numberOfParticles);
    cudaMalloc(&forces, sizeof(float) * 3 * numberOfParticles);
    cudaMemset(forces, 0, sizeof(float) * 3 * numberOfParticles);
    cudaMalloc(&kinetic, sizeof(float));
    cudaMemset(kinetic, 0, sizeof(float));
    cudaMalloc(&potential, sizeof(float));
    cudaMemset(potential, 0, sizeof(float));

    // Initialize global constants
    GlobalConstants local_params;
    local_params.boxSize = boxSize;
    local_params.dudr = dudr;
    local_params.r_cut = r_cut;
    local_params.u_cut = u_cut;
    local_params.delta = delta;
    local_params.numberOfParticles = numberOfParticles;
    local_params.positions = positions;
    local_params.velocities = velocities;
    local_params.forces = forces;
    local_params.kinetic = kinetic;
    local_params.potential = potential;
    cudaMemcpyToSymbol(params, &local_params, sizeof(GlobalConstants));

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "INIT: A CUDA error occured: code=%d, %s, %s\n", errCode, cudaGetErrorName(errCode), cudaGetErrorString(errCode));
        exit(-1);
    }

    int threadsPerBlock = 512;
    int blocks = (numberOfParticles * numberOfParticles + threadsPerBlock - 1) / threadsPerBlock;
    cudaMemset(forces, 0, 3 * numberOfParticles * sizeof(float));
    calculateForceKernel<<<blocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();
    blocks = (numberOfParticles * 2 + threadsPerBlock - 1) / threadsPerBlock;
    calculateBondAngleKernel<<<blocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();

    errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "FORCE1: A CUDA error occured: code=%d, %s, %s\n", errCode, cudaGetErrorName(errCode), cudaGetErrorString(errCode));
        exit(-1);
    }
}

CudaSim::~CudaSim()
{
    cudaFree(positions);
    cudaFree(velocities);
    cudaFree(forces);
}

void CudaSim::advance()
{

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "ADVANCE: A CUDA error occured: code=%d, %s, %s\n", errCode, cudaGetErrorName(errCode), cudaGetErrorString(errCode));
        exit(-1);
    }

    int threadsPerBlock = 512;
    int blocks = (numberOfParticles * 3 + threadsPerBlock - 1) / threadsPerBlock;
    cudaMemset(kinetic, 0, sizeof(float));
    calculateKineticKernel<<<blocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();

    errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "KINETIC: A CUDA error occured: code=%d, %s, %s\n", errCode, cudaGetErrorName(errCode), cudaGetErrorString(errCode));
        exit(-1);
    }

    updateVelocityKernel<<<blocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();

    errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "VELOCITY: A CUDA error occured: code=%d, %s, %s\n", errCode, cudaGetErrorName(errCode), cudaGetErrorString(errCode));
        exit(-1);
    }

    updatePositionKernel<<<blocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();

    errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "POSITION: A CUDA error occured: code=%d, %s, %s\n", errCode, cudaGetErrorName(errCode), cudaGetErrorString(errCode));
        exit(-1);
    }

    blocks = (numberOfParticles * numberOfParticles + threadsPerBlock - 1) / threadsPerBlock;
    cudaMemset(forces, 0, 3 * numberOfParticles * sizeof(float));
    calculateForceKernel<<<blocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();
    blocks = (numberOfParticles * 3 + threadsPerBlock - 1) / threadsPerBlock;
    calculateBondAngleKernel<<<blocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();

    errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "FORCE2: A CUDA error occured: code=%d, %s, %s\n", errCode, cudaGetErrorName(errCode), cudaGetErrorString(errCode));
        exit(-1);
    }

    blocks = (numberOfParticles * 3 + threadsPerBlock - 1) / threadsPerBlock;
    updateVelocityKernel<<<blocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();

    errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "VELOCITY: A CUDA error occured: code=%d, %s, %s\n", errCode, cudaGetErrorName(errCode), cudaGetErrorString(errCode));
        exit(-1);
    }

    calculateKineticKernel<<<blocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();

    errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "KINETIC: A CUDA error occured: code=%d, %s, %s\n", errCode, cudaGetErrorName(errCode), cudaGetErrorString(errCode));
        exit(-1);
    }

}