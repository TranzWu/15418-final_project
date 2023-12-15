//particle.h
#ifndef PARTICLE_H
#define PARTICLE_H

#include <iostream>
#include <vector>



//define particle type

class Particle {
public:
    Particle(int id, char ele, std::vector<double> &position, std::vector<double> &velocity);

    int particleID;
    char element;
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<int> neighborList;    
    
    int getParticleID();

    std::vector<double>& getPosition();

    std::vector<double>& getVelocity();

    void updatePosition();
    
    void updateVeloicity();



// private:
//     int particleID;
//     std::vector<double> position;
//     std::vector<double> velocity;

};








#endif // PARTICLE_H
