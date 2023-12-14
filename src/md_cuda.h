#ifndef MD_CUDA_H
#define MD_CUDA_H

class CudaSim {

private:
    size_t numberOfParticles;

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

public:

    CudaSim(size_t numberOfParticles_in, float boxSize, float dudr,
            float r_cut, float u_cut, float delta, float* init_positions);
    ~CudaSim();
    
    void advance();

    void getPositions(float *out);
    void getVelocities(float *out);
    void getForces(float *out);
    void getKinetic(float *out);
    void getPotential(float *out);
};

#endif // MD_H