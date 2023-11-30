#include "particle.h"
#include "md.h"
#include <iostream>
#include <fstream>



int main(){

    int n = 2;
    double delta = 0.002;
    double t = 0.01;
    int totalSteps = t / delta;

    //add particles to a vector
    std::vector<Particle> particles;

    //initialize particle positions
    for (int i = 0; i < n; i++){
        std::vector<double> p = {i * 1.2, 0, 0};
        std::vector<double> v = {0, 0, 0};
        auto a = Particle(i, p, v);
        particles.push_back(a);
    }



    //Open a file for writing
    std::ofstream outputFile("output.xyz");
    // Check if the file is open
    if (!outputFile.is_open()) {
        std::cerr << "Error opening the file!" << std::endl;
        return 1; // Return an error code
    }

    //initialize force to be 0
    std::vector<std::vector<double>> force(n, std::vector<double>(3, 1));
    double potEng = 0;
    double kinEng = 0;

    calculateForceAndEnergy(particles, force, potEng);


    // for each step, output text to the file
    for (int t = 0; t < totalSteps; t++){
        // do velocity verlet




        //write particle positions to the output files
        outputFile << n << "\n\n";

        for (auto p: particles){
            outputFile << "LJ " << p.position[0] << " " << p.position[1] << " " << p.position[2] << std::endl;
        }


    }
    // Close the file
    outputFile.close();

    return 0;
}
