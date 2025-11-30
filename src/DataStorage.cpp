//
// Created by Oliver Bailey on 28/06/2025.
//

#include "DataStorage.h"
#include "Layer.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

using std::vector;

void DataStorage::saveData(NeuralNetwork& bob, std::string fileName) {
    if (fileName.empty()) {
        std::cout << "Enter name of file:";
        std::cin >> fileName;
    }

    const std::string pathToRemove = "cmake-build-debug";
    std::string path = std::filesystem::current_path().string();
    path = path.substr(0, path.length() - pathToRemove.length());
    path += "data/";

    std::ofstream myFile(path + fileName + ".txt");

    int layerCount = 0;
    vector<Layer> layers =  bob.getLayers();

    for (const Layer& layer : layers) {
        myFile << "Layer " << layerCount << ", Shape: " << layer.getShape() << std::endl;

        saveInfo(myFile, layer);

        layerCount++;

        if (layerCount != layers.size()) myFile << std::endl << std::endl;
    }
}

void DataStorage::saveInfo(std::ofstream& myFile, const Layer& layer) {
    const vector weights = layer.getWeights();

    for (const vector<double>& row : weights) {
        for (double cell : row) {
            myFile << cell << ",";
        }
        myFile << std::endl;
    }
    myFile << std::endl;

    const vector biases = layer.getBiases();

    for (double value : biases) {
        myFile << value << ",";
    }
}

void DataStorage::retrieveData(NeuralNetwork &bob, const std::string& fileName) {
    const std::string pathToRemove = "cmake-build-debug";
    std::string path = std::filesystem::current_path().string();
    path = path.substr(0, path.length() - pathToRemove.length());
    path += "data/";

    std::ifstream myFile(path + fileName + ".txt");

    if (!myFile.good()) {throw std::runtime_error("File not found!");}

    vector layers = bob.getLayers();
    std::string line, value;
    int layerIndex = -1;

    vector<vector<double>> weights;
    vector<double> row;
    vector<double> biases;

    bool settingWeights = true;

    while (std::getline(myFile, line)) {
        if (line.substr(0, 5) == "Layer"){
            if (layerIndex >= 0) {
                layers[layerIndex].setWeights(weights);
                layers[layerIndex].setBiases(biases);
                weights.clear();
                biases.clear();
            }
            layerIndex++;
            continue;}

        if (line.empty()) {
            settingWeights = !settingWeights;
            continue;}

        std::stringstream valuesStream(line);
        if (settingWeights) {
            row.clear();
            while (std::getline(valuesStream, value, ',')) {
                row.push_back(std::stod(value));
            }
            weights.push_back(row);
        } else {
            while (std::getline(valuesStream, value, ',')) {
                biases.push_back(std::stod(value));
            }
        }
    }

    if (layerIndex >= 0) {
        layers[layerIndex].setWeights(weights);
        layers[layerIndex].setBiases(biases);
    }
}