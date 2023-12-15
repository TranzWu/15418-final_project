#ifndef SEQ_SIM_H
#define SEQ_SIM_H

#include "particle.h"
#include "sim.h"

class SeqSim : public Sim {

private:
    size_t number_of_particles;
    size_t timestep;

    float box_size;
    float dudr;
    float r_cut;
    float u_cut;
    float delta;

    float potential;
    float kinetic;

    std::vector<std::vector<float>> forces;
    std::vector<Particle> particles;

    void updateVelocity();
    void updatePosition();
    void calculateKinetic();
    void calculateForceAndEnergy();
    void buildNeighborList();
    void initializeForce();
    void updateBondAndAngle();

    double calculateDistance(Particle p1, Particle p2);

public:
    SeqSim(size_t numberOfParticles_in, float box_size, float dudr,
            float r_cut, float u_cut, float delta, float* init_positions);
    ~SeqSim();

    void advance();

    void getPositions(float *out);
    void getVelocities(float *out);
    float getKinetic();
    float getPotential();
};

#endif // CUDA_SIM_H