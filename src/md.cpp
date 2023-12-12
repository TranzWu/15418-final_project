//md.cpp
#include "md.h"
#include <cmath>

extern int n, N;
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
    for (int i = 0; i < N; i++){
        for (int j = 0; j < 3; j++){
            force[i][j] = 0;
        }
    }
}

void updateBondAndAngle(std::vector<Particle> &particles, std::vector<std::vector<double>> &force, double &potEng){
    double spring = 1000;
    double x0 = 0.8;
    for (int i = 0; i < N; i += 3){
        for (int h = 1; h < 3; h++){
            double dist = calculateDistance(particles[i], particles[i+h]);
            for (int k = 0; k < 3; k++){
                double r = particles[i].position[k] - particles[i+h].position[k];
                if (r < -L/2) r += L;
                if (r > L/2) r -= L;
                potEng += 0.5 * spring * pow(dist - x0, 2);
                //f = spring * (distance - x0)*r/distance
                double f =  -spring * (dist - x0) * r / dist;
                //std::cout << f << std::endl;
                force[i][k] += f;
                force[i+h][k] -= f;
            }
        }
    }
}

void calculateForceAndEnergy(std::vector<Particle> &particles, std::vector<std::vector<double>> &force, double &potEng){

    potEng = 0;
    initializeForce(force);

    for (int i = 0; i < N; i++){
        for (int j = i + 1; j < N; j++){
            if ( !((i % 3 == 0) && (j - i == 1) || (i % 3 == 0) && (j - i == 2) )){
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

    updateBondAndAngle(particles, force, potEng);

}



void updateVelocity(std::vector<Particle> &particles, std::vector<std::vector<double>> &force){
    for (int i = 0; i < N; i++){
        for (int k = 0; k < 3; k++){
            double a = force[i][k] * delta / 2;
            particles[i].velocity[k] += a;
        }
    }
}

void updatePosition(std::vector<Particle> &particles){
    for (int i = 0; i < N; i++){
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
    for (int i = 0; i < N; i++){
        for (int k = 0; k < 3; k++){
            kinEng += 0.5 * std::pow(particles[i].velocity[k], 2);
        }
    }
}
