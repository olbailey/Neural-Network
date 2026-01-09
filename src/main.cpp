#include <iostream>

#include "NeuralNetwork.h"
#include "CsvReader.h"

#include <vector>

using std::vector;

#include "DataStorage.h"

int main() {
	constexpr int BATCH_SIZE = 64;
	constexpr double TRAINING_SPLIT = 0.85;

	auto fileInfo = CsvReader("../data/centralCircleSmall.csv");

	const vector<vector<double>> data = fileInfo.getData();
	const vector<int> labels = fileInfo.getLabels();
	// DataStorage::retrieveData(bob, "delete");

	auto bob = NeuralNetwork(data, labels, { 2, 3, 2 }, 0.03, BATCH_SIZE);

	bob.train(TRAINING_SPLIT);
	// DataStorage::saveData(bob, "");

	return 0;
}