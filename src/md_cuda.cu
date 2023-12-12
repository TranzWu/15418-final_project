#include <cuda.h>

#include "particle.h"
#include "md.h"

struct GlobalConstants {
    size_t numberOfParticles;

    float boxSize;

    double dudr;
    double r_cut;
    double u_cut;
    double delta;
};
__constant__ GlobalConstants cuConstParams;

__device__ float* forces;
__device__ float* positions;
__device__ float* velocities;
__device__ double kinetic;
__device__ double potential;

size_t numberOfParticles;

__device__ double calculateDistance_c(size_t i, size_t j)
{
    float boxSize = cuConstParams.boxSize;

    double distance = 0;
    for (int k = 0; k < 3; k++){
        double diff = positions[i * 3 + k] - positions[j * 3 + k];
        if (diff < -boxSize/2) diff += boxSize;
        if (diff > boxSize/2) diff -= boxSize;
        distance += diff * diff;
    }

    distance = std::sqrt(distance);
    return distance;
}

__global__ void calculateForceKernel()
{
    int numberOfParticles = cuConstParams.numberOfParticles;
    float boxSize = cuConstParams.boxSize;
    double dudr = cuConstParams.dudr;
    double r_cut = cuConstParams.r_cut;
    double u_cut = cuConstParams.u_cut;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index % numberOfParticles;
    int j = index / numberOfParticles;

    if ((i % 3 == 0) && (j - i == 1) || (i % 3 == 0) && (j - i == 2) ) return;

    double dist = calculateDistance_c(i, j);
    if (dist >= r_cut) return;

    double u_actual = 4 * (1/std::pow(dist, 12) - 1/std::pow(dist, 6));
    potential += u_actual - u_cut - (dist - r_cut) * dudr; // TODO: Atomic addd

    for (int k = 0; k < 3; k++)
    {
        double r = positions[i * 3 + k] - positions[j * 3 + k];
        if (r < -boxSize/2) r += boxSize;
        if (r > boxSize/2) r -= boxSize;

        double f = r * (48/pow(dist, 14) - 24/pow(dist, 8) + dudr/dist);
        forces[i * numberOfParticles + k] += f;
        forces[j * numberOfParticles + k] -= f;
    }
}

void calculateForceAndEnergyCuda()
{
    cudaDeviceSynchronize();
    const int threadsPerBlock = 512;
    const int blocks = (numberOfParticles + threadsPerBlock - 1) / threadsPerBlock;

    calculateForceKernel<<<blocks, threadsPerBlock>>>();
}

__global__ void calculateKineticKernel()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > cuConstParams.numberOfParticles * 3) return;
    int i = index % cuConstParams.numberOfParticles;
    int k = index / 3;

    kinetic += 0.5 * pow(velocities[i * 3 + k], 2);
}

void calculateKineticCuda()
{
    cudaDeviceSynchronize();
    const int threadsPerBlock = 512;
    const int blocks = (numberOfParticles + threadsPerBlock - 1) / threadsPerBlock;

    double* energy_v;
    cudaGetSymbolAddress((void **)&energy_v, "kinetic");
    cudaMemset(energy_v, 0, sizeof(double));

    calculateKineticKernel<<<blocks, threadsPerBlock>>>();
}

__global__ void updatePositionKernel()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > cuConstParams.numberOfParticles * 3) return;
    int i = index % cuConstParams.numberOfParticles;
    int k = index / 3;

    positions[i * 3 + k] += velocities[i * 3 + k] * cuConstParams.delta;
    if (positions[i * 3 + k] > cuConstParams.boxSize){
        positions[i * 3 + k] -= cuConstParams.boxSize;
    }
    if (positions[i * 3 + k] < 0){
        positions[i * 3 + k] += cuConstParams.boxSize;
    }
}

void updatePositionCuda()
{
    cudaDeviceSynchronize();
    const int threadsPerBlock = 512;
    const int blocks = (numberOfParticles + threadsPerBlock - 1) / threadsPerBlock;

    updatePositionKernel<<<blocks, threadsPerBlock>>>();
}

__global__ void updateVelocityKernel()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > cuConstParams.numberOfParticles * 3) return;
    int i = index % cuConstParams.numberOfParticles;
    int k = index / 3;


    double a = forces[i * 3 + k] * cuConstParams.delta;
    velocities[i * 3 + k] += a;
}

void updateVelocityCuda()
{
    cudaDeviceSynchronize();
    const int threadsPerBlock = 512;
    const int blocks = (numberOfParticles + threadsPerBlock - 1) / threadsPerBlock;

    updateVelocityKernel<<<blocks, threadsPerBlock>>>();
}

void getPositions(float *out)
{
    printf("Getting positions from CUDA device\n");
    // Copies positions into parameter from device
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(out, positions, 3 * numberOfParticles * sizeof(float));
}

void getKinetic(double *out)
{
    // Copies kinetic energy from device to out
    cudaMemcpyFromSymbol(out, kinetic, sizeof(float));
}

void getPotential(double *out)
{
    // Copies potential energy from device to out
    cudaMemcpyFromSymbol(out, potential, sizeof(float));
}

void initializeCuda(size_t numberOfParticles_in, float boxSize, double dudr,
                double r_cut, double u_cut, double delta, float* init_positions)
{
    printf("Initializing CUDA\n");
    // Initialize Global constant parameters
    GlobalConstants params;
    params.boxSize = boxSize;
    params.boxSize = dudr;
    params.boxSize = boxSize;
    params.r_cut = r_cut;
    params.u_cut = u_cut;
    params.delta = delta;
    params.numberOfParticles = numberOfParticles_in;
    numberOfParticles = numberOfParticles_in;
    cudaMemcpyToSymbol(cuConstParams, &params, sizeof(GlobalConstants));

    // Initialize kinetic and potential
    double* energy_v;
    cudaGetSymbolAddress((void **)&energy_v, "kinetic");
    cudaMemset(energy_v, 0, sizeof(double));
    cudaGetSymbolAddress((void **)&energy_v, "potential");
    cudaMemset(energy_v, 0, sizeof(double));

    // Initialize positions, forces, and velocities
    cudaMemcpyToSymbol(positions, &init_positions, 3 * numberOfParticles * sizeof(float));
    float* device_v;
    cudaGetSymbolAddress((void **)&device_v, "forces");
    cudaMemset(device_v, 0, 3 * numberOfParticles * sizeof(double));
    cudaGetSymbolAddress((void **)&device_v, "velocities");
    cudaMemset(device_v, 0, 3 * numberOfParticles * sizeof(double));
}