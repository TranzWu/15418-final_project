//md.cpp
#include "md.h"
#include <cmath>

extern int n;
extern double delta, L, r_cut, u_cut, dudr;



double calculateDistance(Particle p1, Particle p2){
    double distance = 0;
    for (int i = 0; i < 3; i++){
        double diff = p1.position[i] - p2.position[i];
        if (diff < -L/2) diff += L;
        if (diff > L/2) diff -= L;
        distance += diff * diff;
    }

    distance = std::sqrt(distance);
    return distance;
}

void initializeForce(std::vector<std::vector<double>> &force){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < 3; j++){
            force[i][j] = 0;
        }
    }
}

void calculateForceAndEnergy(std::vector<Particle> &particles, std::vector<std::vector<double>> &force, double &potEng){

    potEng = 0;
    initializeForce(force);


    for (int i = 0; i < n; i++){
        for (int j = i + 1; j < n; j++){
            double dist = calculateDistance(particles[i], particles[j]);
            //calculate potential energy for each pair
            if (dist < r_cut){
                double u_actual = 4 * (1/std::pow(dist, 12) - 1/std::pow(dist, 6));
                potEng += u_actual - u_cut - (dist - r_cut) * dudr;
                //std::cout << dist << std::endl;
                for (int k = 0; k < 3; k++){
                    double r = particles[i].position[k] - particles[j].position[k];
                    if (r < -L/2) r += L;
                    if (r > L/2) r -= L;
                    
                    double f = r * (48/std::pow(dist, 14) - 24/std::pow(dist, 8) + dudr/dist);
                    //std::cout << "f is " << f << std::endl;
                    force[i][k] += f;
                    force[j][k] -= f;
                }
            }

        }
    }

}



void updateVelocity(std::vector<Particle> &particles, std::vector<std::vector<double>> &force){
    for (int i = 0; i < n; i++){
        for (int k = 0; k < 3; k++){
            particles[i].velocity[k] += force[i][k] * delta / 2;
        }
    }
}

void updatePosition(std::vector<Particle> &particles){
    for (int i = 0; i < n; i++){
        for (int k = 0; k < 3; k++){
            particles[i].position[k] += particles[i].velocity[k] * delta;
            if (particles[i].position[k] > L){
                particles[i].position[k] -= L;
            }
            if (particles[i].position[k] < 0){
                particles[i].position[k] += L;
            }
        }
    }
}



void calculateKenetic(std::vector<Particle> &particles, double &kinEng){
    kinEng = 0;
    for (int i = 0; i < n; i++){
        for (int k = 0; k < 3; k++){
            kinEng += 0.5 * std::pow(particles[i].velocity[k], 2);
        }
    }
}
