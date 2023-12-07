//particle.cpp
#include "particle.h"


Particle::Particle(int id, char ele, std::vector<double>& pos, std::vector<double>& vel):particleID(id), element(ele), position(pos), velocity(vel) {
    
}

int Particle::getParticleID(){
    return particleID;
}

std::vector<double>& Particle::getPosition(){
    return position;
}

std::vector<double>& Particle::getVelocity(){
    return velocity;
}