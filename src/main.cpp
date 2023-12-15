//main.cpp
#include "particle.h"
#include "md.h"
#include "sim.h"
#include "seq_sim.h"
#include "cuda_sim.h"
#include "omp_sim.h"
#include "cycleTimer.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <utility>
#include <chrono>
#include <getopt.h>
#include <cstring>

static int debug_flag;

// 0 for sequential, 1 for CUDA, 2 for ISPC
int backend = 0;

// #include <omp.h>


double L = 11.5;
int n = 6 * 6 * 6;
int N = n * 3;
double delta = 0.002;
double t = 0.1;
int totalSteps = t / delta;

//define cutoff radius to save computational time
double r_cut = 2.5;
double u_cut = 4 * (1/std::pow(r_cut, 12) - 1/std::pow(r_cut, 6));
//dudr = -(48/r_cut**13 - 24/r_cut**7)
double dudr = -(48/std::pow(r_cut, 13) - 24/std::pow(r_cut, 7));

int sim_main(size_t dim, int backend)
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
    std::ofstream outputFile("output.xyz");
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
    Sim* sim;
    if (backend == 0)
        sim = new SeqSim(numParticles, L, dudr, r_cut, u_cut, delta, positions);
    else if (backend == 1)
        sim = new CudaSim(numParticles, L, dudr, r_cut, u_cut, delta, positions);
    else
        sim = new SeqSim(numParticles, L, dudr, r_cut, u_cut, delta, positions);

    // Begin simulation
    for (int t = 0; t < totalSteps; t++){
        sim->advance();

        if (debug_flag)
        {
            float kinetic = sim->getKinetic();
            float potential = sim->getPotential();
            std::cout << kinetic << " " << potential << " " << kinetic + potential << std::endl;
        }

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
    printf("  -d  --debug            Turn on debug \n");
    printf("  -?  --help             This message\n");
}

int main(int argc, char** argv){

    int opt;
    static struct option long_options[] = {
        {"backend", required_argument, NULL, 'b'},
        {"debug", optional_argument, NULL, 'd'},
        {NULL, 0, NULL, 0}
    };

    while ((opt = getopt_long(argc, argv, "d?b:", long_options, NULL)) != EOF) {
        switch (opt) {
        case 'b':
            backend = atoi(optarg);
            break;
        case 'd':
            debug_flag = true;
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }

    float startTime = CycleTimer::currentSeconds();
    int res = sim_main(std::cbrt(n), backend);
    float endTime = CycleTimer::currentSeconds();

    printf("Runtime = %f\n", endTime - startTime);

    return res;
}
