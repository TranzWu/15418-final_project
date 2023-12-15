//main.cpp
#include "particle.h"
#include "md.h"
#include "md_cuda.h"
#include "cycleTimer.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <utility>
#include <getopt.h>
#include <cstring>


// 0 for sequential, 1 for CUDA, 2 for ISPC
int backend = 0;

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

int sequential_main()
{
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

int cuda_main(size_t dim)
{
    // Initialize positions and velocities
    float positions[dim * dim * dim * 3 * 3];
    float velocities[dim * dim * dim * 3 * 3];
    memset(velocities, 0, dim * dim * dim * 3 * sizeof(float));
    for (int m = 0; m < dim; m++){
        for (int l = 0; l < dim; l++){
            for (int q = 0; q < dim; q++){
                // Oxygen atom
                float *pos = &positions[m * dim * dim * 3 * 3 + l * dim * 3 * 3 + q * 3 * 3];
                pos[0] = m * 2;
                pos[1] = l * 2;
                pos[2] = q * 2;

                // First hydrogen atom
                pos = &positions[m * dim * dim * 3 * 3 + l * dim * 3 * 3 + q * 3 * 3 + 3];
                pos[0] = m * 2 + 0.5;
                pos[1] = l * 2;
                pos[2] = q * 2;

                // Second hydrogen atom
                pos = &positions[m * dim * dim * 3 * 3 + l * dim * 3 * 3 + q * 3 * 3 + 6];
                pos[0] = m * 2;
                pos[1] = l * 2 + 0.5;
                pos[2] = q * 2;
            }
        }
    }

    // Open output file
    std::ofstream outputFile("cuda_output.xyz");
    if (!outputFile.is_open()) {
        std::cerr << "Error opening the file!" << std::endl;
        return 1;
    }

    // Write initial positions out to file
    size_t numParticles = dim * dim * dim * 3;
    outputFile << N << "\n\n";
    for (size_t i = 0; i < numParticles; i++)
    {
        char element;
        if (i % 3 == 0)
            element = 'O';
        else
            element = 'H';
        outputFile << element << " " << " " << positions[i * 3] << " " << positions[i * 3 + 1] << " " << positions[i * 3 + 2] \
                   << " " << velocities[i * 3] << " " << velocities[i * 3 + 1] << " " << velocities[i * 3 + 2] << std::endl;
    }

    // Initialize CUDA simulator
    CudaSim* sim = new CudaSim(numParticles, L, dudr, r_cut, u_cut, delta, positions);

    // Begin simulation
    for (int t = 0; t < totalSteps; t++){
        sim->advance();

        float kinetic, potential;
        sim->getKinetic(&kinetic);
        sim->getPotential(&potential);

        // std::cout << kinetic << " " << potential << " " << kinetic + potential << std::endl;

        // Write new positions and velocities to output file
        if (t % 20 == 0){
            sim->getPositions(positions);
            sim->getVelocities(velocities);
            outputFile << N << "\n\n";
            for (size_t i = 0; i < numParticles; i++)
            {
                char element;
                if (i % 3 == 0) element = 'O';
                else element = 'H';
                outputFile << element << " " << positions[i * 3] << " " << positions[i * 3 + 1] << " " << positions[i * 3 + 2] \
                           << " " << velocities[i * 3] << " " << velocities[i * 3 + 1] << " " << velocities[i * 3 + 2] << std::endl;
            }
        }


    }

    outputFile.close();

    return 0;
}

void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -b  --backend <INT>    Select backend (0 : Sequential, 1 : CUDA) \n");
    printf("  -?  --help             This message\n");
}

int main(int argc, char** argv){

    int opt;
    static struct option long_options[] = {
        {"backend", required_argument, NULL, 'b'},
        {NULL, 0, NULL, 0}
    };

    while ((opt = getopt_long(argc, argv, "?b:", long_options, NULL)) != EOF) {

        switch (opt) {
        case 'b':
            backend = atoi(optarg);
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }


    float startTime = CycleTimer::currentSeconds();
    int res;
    if (backend == 0)
    {
        res = sequential_main();
    }
    else if (backend == 1)
    {
        res = cuda_main(std::cbrt(n));
    }
    else
    {
        printf("ERROR Unimplemented\n");
        return 1;
    }
    float endTime = CycleTimer::currentSeconds();

    printf("Runtime = %f\n", endTime - startTime);

    return res;
}
