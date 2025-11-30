//
// Created by Oliver Bailey on 30/06/2025.
//

#ifndef ADAMLAYER_H
#define ADAMLAYER_H

#include <vector>

class AdamLayer {
    std::vector<std::vector<double>> m_weights;
    std::vector<std::vector<double>> v_weights;
    std::vector<double> m_biases;
    std::vector<double> v_biases;
};

#endif //ADAMLAYER_H