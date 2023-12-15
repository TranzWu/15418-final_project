#ifndef SIM_H
#define SIM_H

class Sim {
public:
    virtual void advance()=0;

    virtual void getPositions(float *out)=0;
    virtual void getVelocities(float *out)=0;
    virtual float getKinetic()=0;
    virtual float getPotential()=0;
};

#endif // SIM_H