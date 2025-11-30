//
// Created by olbai on 28/06/2025.
//

#ifndef NUMPTYHELPER_H
#define NUMPTYHELPER_H

#include <vector>
#include <iostream>

using std::vector;

namespace  NumptyHelper {
    template <typename T>
    void print2D(const vector<vector<T>>& matrix) {
        for (vector<T> row : matrix) {
            for (T cell : row) {
                std::cout << cell << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    template <typename T>
    void print1D(const vector<T>& vec) {
        for (T value : vec) {
            std::cout << value << ", ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }
};

#endif //NUMPTYHELPER_H
