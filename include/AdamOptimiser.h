//
// Created by Oliver Bailey on 30/06/2025.
//

#ifndef ADAMOPTIMISER_H
#define ADAMOPTIMISER_H

#include "AdamLayer.h"
#include <vector>

class AdamOptimiser {
    double learningRate;
    double beta1;
    double beta2;
    double epsilon;

    std::vector<AdamLayer> layers;

public:
    AdamOptimiser();
};

#endif //ADAMOPTIMISER_H
