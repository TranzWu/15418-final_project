#include "md.h"
#include <cmath>



double calculateDistance(Particle p1, Particle p2){
    double distance = 0;
    for (int i = 0; i < 3; i++){
        double diff = p1.position[i] - p2.position[i];
        distance += diff * diff;
    }

    distance = std::sqrt(distance);
    return distance;
}

void calculateForceAndEnergy(std::vector<Particle> &particles, std::vector<std::vector<double>> &force, double &potEng){
    int n = particles.size();

    for (int i = 0; i < n; i++){
        for (int j = i + 1; j < n; j++){
            double dist = calculateDistance(particles[i], particles[j]);
            
            for (int k = 0; k < 3; k++){
                ;
            }

        }
    }
}