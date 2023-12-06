//main.cpp
#include "particle.h"
#include "md.h"
#include <iostream>
#include <fstream>
#include <cmath>

double L = 6.5;
int n = 6 * 6 * 6;
double delta = 0.002;
double t = 100;
int totalSteps = t / delta;

//define cutoff radius to save computational time
double r_cut = 2.5;
double u_cut = 4 * (1/std::pow(r_cut, 12) - 1/std::pow(r_cut, 6));
//dudr = -(48/r_cut**13 - 24/r_cut**7)
double dudr = -(48/std::pow(r_cut, 13) - 24/std::pow(r_cut, 7));



int main(){


    //add particles to a vector
    std::vector<Particle> particles;

    //initialize particle positions
    
    
    int row = std::cbrt(n);
    for (int m = 0; m < row; m++){
        for (int l = 0; l < row; l++){
            for (int q = 0; q < row; q++){
                std::vector<double> p(3);
                p[0] = m * 1;
                p[1] = l * 1;
                p[2] = q * 1;

                std::vector<double> v = {0, 0, 0};
                auto a = Particle(m * row + l * row + q, p, v);
                particles.push_back(a);
            }
        }
    }


    //Open a file for writing
    std::ofstream outputFile("output.xyz");
    // Check if the file is open
    if (!outputFile.is_open()) {
        std::cerr << "Error opening the file!" << std::endl;
        return 1; // Return an error code
    }

    //initialize force to be 0
    std::vector<std::vector<double>> force(n, std::vector<double>(3, 0));
    double potEng = 0;
    double kinEng = 0;

    calculateForceAndEnergy(particles, force, potEng);


    // for each step, output text to the file
    for (int t = 0; t < totalSteps; t++){
        // do velocity verlet
        updateVelocity(particles, force);
        updatePosition(particles);
        calculateForceAndEnergy(particles, force, potEng);
        updateVelocity(particles, force);
        calculateKenetic(particles, kinEng);

        //std::cout << kinEng << " " << potEng << " " << kinEng + potEng << std::endl;
        std::cout << (double)t/totalSteps * 100 << "%" << std::endl;

        //write particle positions to the output files
        outputFile << n << "\n\n";

        for (auto p: particles){
            outputFile << "LJ " << p.position[0] << " " << p.position[1] << " " << p.position[2] << " " <<p.velocity[0] << " " << p.velocity[1] << " " << p.velocity[2]  << std::endl;
        }


    }
    // Close the file
    outputFile.close();

    return 0;
}
