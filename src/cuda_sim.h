#ifndef CUDA_SIM_H
#define CUDA_SIM_H

#include "sim.h"

class CudaSim : public Sim {

private:
    size_t numberOfParticles;
    size_t timesteps;

    float boxSize;

    float dudr;
    float r_cut;
    float u_cut;
    float delta;

    float* forces;
    float* positions;
    float* velocities;

    float* kinetic;
    float* potential;

    short* neighbors;
    short* counts;

public:
    CudaSim(size_t numberOfParticles_in, float boxSize, float dudr,
            float r_cut, float u_cut, float delta, float* init_positions);
    ~CudaSim();

    void advance();

    void getPositions(float *out);
    void getVelocities(float *out);
    float getKinetic();
    float getPotential();
};

#endif // CUDA_SIM_H