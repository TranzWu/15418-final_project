#include <iostream>
#include <vector>


//define particle type

class Particle {
public:
    Particle(int id, std::vector<double> &position, std::vector<double> &velocity);

    int particleID;
    std::vector<double> position;
    std::vector<double> velocity;    
    
    int getParticleID();

    std::vector<double>& getPosition();

    std::vector<double>& getVelocity();

    void updatePosition();
    
    void updateVeloicity();



// private:
//     int particleID;
//     std::vector<double> position;
//     std::vector<double> velocity;

};