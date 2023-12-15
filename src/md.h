#include "particle.h"
#ifndef MD_H
#define MD_H


void initializeForce(std::vector<std::vector<double>> &force);

void updateBondAndAngle(std::vector<Particle> &particles, std::vector<std::vector<double>> &force, double &potEng);

void calculateForceAndEnergy(std::vector<Particle> &particles, std::vector<std::vector<double>> &force, double &potEng);

void updateVelocity(std::vector<Particle> &particles, std::vector<std::vector<double>> &force);


void updatePosition(std::vector<Particle> &particles);

void calculateKenetic(std::vector<Particle> &particles, double &kinEng);

void buildNeighborList(std::vector<Particle> &particles);


#endif // MD_H