//main.cpp
#include "particle.h"
#include "md.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <utility>

double L = 12;
int n = 5 * 5 * 5;
int N = n * 3;
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
    int count = 0;
    for (int m = 0; m < row; m++){
        for (int l = 0; l < row; l++){
            for (int q = 0; q < row; q++){
                std::vector<double> p_o(3);  //create Oxygen atom
                p_o[0] = m * 2;
                p_o[1] = l * 2;
                p_o[2] = q * 2;

                std::vector<double> v_o = {0, 0, 0};
                auto a1 = Particle((m * row + l * row + q) * 3, 'O', p_o, v_o);
                particles.push_back(a1);
                


                std::vector<double> p_h1(3);  //create the first hydrogen atom
                p_h1[0] = m * 2 + 0.5;
                p_h1[1] = l * 2;
                p_h1[2] = q * 2;


                std::vector<double> v_h1 = {0, 0, 0};
                auto a2 = Particle((m * row + l * row + q) * 3 + 1, 'H', p_h1, v_h1);
                particles.push_back(a2);
                


                std::vector<double> p_h2(3);  //create the second hydrogen atom
                p_h2[0] = m * 2;
                p_h2[1] = l * 2 + 0.5;
                p_h2[2] = q * 2;

                std::vector<double> v_h2 = {0, 0, 0};
                auto a3 = Particle((m * row + l * row + q) * 3 + 2, 'H', p_h2, v_h2);
                particles.push_back(a3);
                
                

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

    //write the initial position to file
    outputFile << N << "\n\n";

    for (auto &p: particles){
        outputFile << p.element << " " << p.position[0] << " " << p.position[1] << " " << p.position[2] << " " <<p.velocity[0] << " " << p.velocity[1] << " " << p.velocity[2]  << std::endl;
    }

    //initialize force to be 0
    std::vector<std::vector<double>> force(N, std::vector<double>(3, 0));
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

        std::cout << kinEng << " " << potEng << " " << kinEng + potEng << std::endl;
        //std::cout << (double)t/totalSteps * 100 << "%" << std::endl;

        //write particle positions to the output files
        if (t % 20 == 0){
            outputFile << N << "\n\n";

            for (auto p: particles){
                //std::cout << p.element << std::endl;
                outputFile << p.element << " " << p.position[0] << " " << p.position[1] << " " << p.position[2] << " " <<p.velocity[0] << " " << p.velocity[1] << " " << p.velocity[2]  << std::endl;
            }
        }


    }
    // Close the file
    outputFile.close();

    return 0;
}
