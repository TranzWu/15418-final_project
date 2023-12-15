#include "md.h"
#include "omp_sim.h"

#include <cmath>
#include <cstring>

// extern int n, N;
// extern double delta, L, r_cut, u_cut, dudr;

double OmpSim::calculateDistance(Particle p1, Particle p2){
    double distance = 0;
    for (int i = 0; i < 3; i++){
        double diff = p1.position[i] - p2.position[i];
        if (diff < -box_size/2) diff += box_size;
        if (diff > box_size/2) diff -= box_size;
        distance += diff * diff;
    }

    distance = std::sqrt(distance);
    return distance;
}

void OmpSim::initializeForce(){
    for (int i = 0; i < number_of_particles; i++){
        for (int j = 0; j < 3; j++){
            forces[i][j] = 0;
        }
    }
}

void OmpSim::updateBondAndAngle()
{
    double spring = 1000;
    double x0 = 0.8;
    #pragma omp parallel for
    for (int i = 0; i < number_of_particles; i += 3){
        for (int h = 1; h < 3; h++){
            double dist = calculateDistance(particles[i], particles[i+h]);
            for (int k = 0; k < 3; k++){
                double r = particles[i].position[k] - particles[i+h].position[k];
                if (r < -box_size/2) r += box_size;
                if (r > box_size/2) r -= box_size;
                potential += 0.5 * spring * pow(dist - x0, 2);
                double f =  -spring * (dist - x0) * r / dist;
                forces[i][k] += f;
                forces[i+h][k] -= f;
            }
        }
    }
}


void OmpSim::buildNeighborList(){
    double radius = 3;

    for (int i = 0; i < number_of_particles; i++){
        particles[i].neighborList.clear(); //first clean the neighbot list
        for (int j = i + 1; j < number_of_particles; j++){
            if ( !((i % 3 == 0) && (j - i == 1) || (i % 3 == 0) && (j - i == 2) )){
                double dist = calculateDistance(particles[i], particles[j]);
                if (dist < radius){
                    particles[i].neighborList.push_back(j);
                }
            }

        }
    }

}

void OmpSim::calculateForceAndEnergy()
{
    potential = 0;
    initializeForce();

    #pragma omp parallel for
    for (int i = 0; i < number_of_particles; i++){
        for (int j: particles[i].neighborList){
            double dist = calculateDistance(particles[i], particles[j]);
                double u_actual = 4 * (1/std::pow(dist, 12) - 1/std::pow(dist, 6));
                potential += u_actual - u_cut - (dist - r_cut) * dudr;
                for (int k = 0; k < 3; k++){
                    double r = particles[i].position[k] - particles[j].position[k];
                    if (r < -box_size/2) r += box_size;
                    if (r > box_size/2) r -= box_size;

                    double f = r * (48/std::pow(dist, 14) - 24/std::pow(dist, 8) + dudr/dist);
                    forces[i][k] += f;
                    forces[j][k] -= f;
                }
        }
    }

    updateBondAndAngle();
}



void OmpSim::updateVelocity()
{
    #pragma omp parallel for
    for (int i = 0; i < number_of_particles; i++){
        for (int k = 0; k < 3; k++){
            double a = forces[i][k] * delta / 2;
            particles[i].velocity[k] += a;
        }
    }
}

void OmpSim::updatePosition()
{
    #pragma omp parallel for
    for (int i = 0; i < number_of_particles; i++){
        for (int k = 0; k < 3; k++){
            particles[i].position[k] += particles[i].velocity[k] * delta;
            if (particles[i].position[k] > box_size){
                particles[i].position[k] -= box_size;
            }
            if (particles[i].position[k] < 0){
                particles[i].position[k] += box_size;
            }
        }
    }
}

void OmpSim::calculateKinetic(){
    kinetic = 0;
    for (int i = 0; i < number_of_particles; i++){
        for (int k = 0; k < 3; k++){
            kinetic += 0.5 * std::pow(particles[i].velocity[k], 2);
        }
    }
}

OmpSim::OmpSim(size_t number_of_particles_in, float box_size, float dudr,
               float r_cut, float u_cut, float delta, float* init_positions)
{
    for (int i = 0; i < number_of_particles_in; i++)
    {
        char ele;
        if (i % 3 == 0) ele = 'O';
        else ele = 'H';

        std::vector<double> position {init_positions[i * 3],
                                 init_positions[i * 3 + 1],
                                 init_positions[i * 3 + 2]};
        std::vector<double> velocity {0, 0, 0};

        particles.push_back(Particle(i, ele, position, velocity));

        std::vector<float> force {0, 0, 0};
        forces.push_back(force);
    }

    kinetic = 0;
    potential = 0;
    number_of_particles = number_of_particles_in;
    box_size = box_size;
    dudr = dudr;
    r_cut = r_cut;
    u_cut = u_cut;
    delta = delta;

    buildNeighborList();
    calculateForceAndEnergy();
}

OmpSim::~OmpSim() {}

void OmpSim::advance()
{
    if (timestep % 50 == 0) buildNeighborList();
    updateVelocity();
    updatePosition();
    calculateForceAndEnergy();
    updateVelocity();
    calculateKinetic();

    timestep += 1;
}

void OmpSim::getPositions(float *out)
{
    for (int i = 0; i < number_of_particles; i++)
    {
        memcpy(&out[i * 3], &particles[i].position[0], 3 * sizeof(float));
    }
}

void OmpSim::getVelocities(float *out)
{
    for (int i = 0; i < number_of_particles; i++)
    {
        std::memcpy(&out[i * 3], &particles[i].velocity[0], 3 * sizeof(float));
    }
}

float OmpSim::getKinetic()
{
    return kinetic;
}

float OmpSim::getPotential()
{
    return potential;
}

