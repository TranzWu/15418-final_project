#include "particle.h"
#ifndef MD_CUDA_H
#define MD_CUDA_H

void calculateForceAndEnergyCuda();

void updateVelocityCuda();

void updatePositionCuda();

void calculateKineticCuda();

void initializeCuda(size_t numberOfParticles_in, float boxSize, double dudr,
                    double r_cut, double u_cut, double delta, float* init_positions);

void getPositions(float *out);

void getKinetic(double *out);

void getPotential(double *out);

#endif // MD_H