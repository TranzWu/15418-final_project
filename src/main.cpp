//main.cpp
#include "particle.h"
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
#include <pthread.h>
#include <sstream>

static int debug_flag;

// 0 for sequential, 1 for CUDA, 2 for ISPC
int backend = 0;

double L = 11.5;
int d = 6;
int n = d * d * d;
int N = n * 3;
double delta = 0.002;
double t = 100;
int totalSteps = t / delta;

//define cutoff radius to save computational time
double r_cut = 2.5;
double u_cut = 4 * (1/std::pow(r_cut, 12) - 1/std::pow(r_cut, 6));
double dudr = -(48/std::pow(r_cut, 13) - 24/std::pow(r_cut, 7));

float th_param[6 * 6 * 6 * 3 * 3 * 2];

bool outputting = false;
size_t timestep = 0;

void *write_output(void* ptr)
{
    float* positions = (float *)th_param;
    float* velocities = &positions[N];

    std::stringstream file_title;
    file_title << "data/output" << timestep << ".xyz";

    // Open output file
    std::ofstream outputFile(file_title.str());
    timestep += 1;
    if (!outputFile.is_open()) {
        std::cerr << "Error opening the file!" << std::endl;
        return nullptr;
    }

    outputFile << N << "\n\n";
    for (size_t i = 0; i < N; i++)
    {
        char element;
        if (i % 3 == 0) element = 'O';
        else element = 'H';
        outputFile << element << " " << positions[i * 3] << " " << positions[i * 3 + 1] << " " << positions[i * 3 + 2] \
                    << " " << velocities[i * 3] << " " << velocities[i * 3 + 1] << " " << velocities[i * 3 + 2] << std::endl;
    }

    return nullptr;
}

int sim_main(size_t dim, int backend)
{
    // Initialize positions and velocities
    float* positions = &th_param[0];
    float* velocities = &th_param[dim * dim * dim * 3 * 3];
    memset(velocities, 0, sizeof(velocities));
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

    // Write initial positions out to file
    size_t numParticles = dim * dim * dim * 3;
    std::ofstream outputFile("outputfile.xyz");
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
    outputFile.close();

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
            pthread_t output_thread;
            sim->getPositions(positions);
            sim->getVelocities(velocities);

            pthread_create(&output_thread, NULL, write_output, nullptr);
            pthread_detach(output_thread);
        }
    }
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
    int res = sim_main(d, backend);
    float endTime = CycleTimer::currentSeconds();

    printf("Runtime = %f\n", endTime - startTime);

    return res;
}
