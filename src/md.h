#include "particle.h"
#ifndef MD_H
#define MD_H


double calculateDistance(Particle p1, Particle p2);

void calculateForceAndEnergy(std::vector<Particle> &particles, std::vector<std::vector<double>> &force, double &potEng);

void initializeForce(std::vector<std::vector<double>> &force);

void updateBondAndAngle(std::vector<Particle> &particles, std::vector<std::vector<double>> &force, double &potEng);

void updateVelocity(std::vector<Particle> &particles, std::vector<std::vector<double>> &force);

void updatePosition(std::vector<Particle> &particles);

void calculateKenetic(std::vector<Particle> &particles, double &kinEng);


#endif // MD_H