#include "types.h"



int main(){

    std::vector<double> v1 = {1,2,3};
    std::vector<double> v2 = {4,5,6};

    auto p1 = Particle(5, v1, v2);

    std::cout << p1.particleID << std::endl;

    // auto temp = p1.getPosition();

    // for (auto i: temp){
    //     std::cout << i << " ";
    // }

    // std::cout << std::endl;

    return 0;
}

