//
// Created by Oliver Bailey on 28/06/2025.
//

#ifndef SAVINGDATA_H
#define SAVINGDATA_H

#include <NeuralNetwork.h>
#include <Layer.h>

#include <fstream>

class DataStorage {
public:
    static void saveData(NeuralNetwork &bob, std::string fileName);
    static void retrieveData(NeuralNetwork &bob, const std::string &fileName);

private:
    static void saveInfo(std::ofstream &myFile, const Layer &layer);
};

#endif //SAVINGDATA_H
