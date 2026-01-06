#include <iostream>

#include "NeuralNetwork.h"
#include "Layer.h"
#include "Numpty.h"
#include "DataStorage.h"

#include <vector>
#include <cmath>
#include <ctime>
#include <thread>
#include <random>

#include "NumptyHelper.h"

using std::vector;

NeuralNetwork::NeuralNetwork(const vector<vector<double>>& data, const vector<int>& labels, vector<int> shape, const double learningRate)
	: size(shape.size() - 1), learningRate(learningRate), data(data), labels(labels), N(labels.size())
{
	for (size_t i = 0; i < size; i++) {
		layers.emplace_back(shape[i], shape[i + 1]); // constructs Layer object within the push
	}
}

void NeuralNetwork::train(const int batchSize, const double trainingSplit) {
	loadData(trainingSplit);

	clock_t beginTime = -30000;
	double previousCost = 1000;
	bool stopped = false;

	// Threads operation
	std::thread terminateThread([this, &stopped] {
			threadExample(stopped);
		});

	while (!stopped) {
		// Train neural network
		for (int i = 1; i < N / batchSize; i++) {
			const size_t startPoint = batchSize * (i - 1);
			size_t endPoint = batchSize * i;

			if (endPoint > trainingData.size())
				endPoint = trainingData.size();

			const vector<vector<double>> trainingBatch = Numpty::deepCopy2D(trainingData, startPoint, endPoint);
			const vector<int> labelsBatch = Numpty::copy(trainingLabels, startPoint, endPoint);

			const vector<vector<double>> predicts = forwardPass(trainingBatch);
			const double lossValue = loss(predicts, labelsBatch);
			backpropagation(trainingBatch, labelsBatch);
		}
		if (iterations % 1000 == 0) {
			std::cout << '\n' << "Iterations: " << iterations << '\n';
			auto values = forwardPass(trainingData);
			std::cout << values[0][0] << '\n';
			std::cout << "Training loss: " << loss(values, trainingLabels) << '\n';
			std::cout << "Testing loss: " << loss(forwardPass(testingData), testingLabels) << '\n';
		}

		// if (clock() - beginTime > 30000 || iterations % 1000 == 0) {
		// 	beginTime = clock();
		// 	const double costValue = loss(forwardPass(trainingData), trainingLabels);
		//
		// 	if (previousCost - costValue < 0.0001) {
		// 		stopped = true;
		// 	}
		//
		// 	previousCost = costValue;
		// 	std::cout << costValue << std::endl;
		//
		// 	const double accuracy = cost(forwardPass(testingData), testingLabels);
		// 	std::cout << accuracy << "/" << testingData.size() << '\n';
		// }
		iterations++;
	}
	std::cout << loss(trainingData, trainingLabels) << std::endl;
	DataStorage::saveData(*this, "");

	if (terminateThread.joinable()) {
		terminateThread.join();
	}
}

vector<vector<double>> NeuralNetwork::forwardPass(const vector<vector<double>>& inputs) const {
	vector outputs(inputs.size(), vector<double>(2));
	for (size_t i = 0; i < inputs.size(); i++) {
		outputs[i] = layers[0].calculateLayerOutput(inputs[i], "Tanh");
		for (size_t j = 1; j < size - 1; j++) {
			outputs[i] = layers[j].calculateLayerOutput(outputs[i], "Tanh");
		}
		outputs[i] = layers[size - 1].calculateLayerOutput(outputs[i], "Softmax");
	}

	return outputs;
}

double NeuralNetwork::loss(const vector<vector<double>>& predicts, const vector<int>& expectedOutputs) {
	const vector<int> expectedCopy = Numpty::copy(expectedOutputs, 0, expectedOutputs.size());
	const vector<int> correctedLabels = Numpty::subtractScalar(expectedCopy, 1);

	vector<double> losses = Numpty::logarithm(predicts, correctedLabels);
	Layer::checkInvalidNum(losses[0]);

	Numpty::multiplyByScalar(losses, -1);

	return Numpty::mean(losses);
}

void NeuralNetwork::backpropagation(const vector<vector<double>>& trainingData, const vector<int> &trainingLabels) {

}

double NeuralNetwork::cost(const vector<vector<double> > &predicts, const vector<int> &expectedOutputs) {
	vector<int> predictedOutputChoices = Numpty::argmax(predicts);
	Numpty::addScalar(predictedOutputChoices, 1);

	return Numpty::equal(predictedOutputChoices, expectedOutputs);
}

void NeuralNetwork::loadData(const double trainingSplit) {

	const vector<size_t>& randomIndices = randomisedIndexes();
	const auto randomData = randomiseVector(data, randomIndices);
	const auto randomLabels = randomiseVector(labels, randomIndices);

	const long testingCutOff = static_cast<long>(std::floor(N * trainingSplit));

	trainingData = Numpty::deepCopy2D(randomData, 0, testingCutOff);
	trainingLabels = Numpty::copy(randomLabels, 0, testingCutOff);

	testingData = Numpty::deepCopy2D(randomData, testingCutOff, N);
	testingLabels = Numpty::copy(randomLabels, testingCutOff, N);
}

vector<size_t> NeuralNetwork::randomisedIndexes() const {
	vector<size_t> randomValues(N);
	std::random_device dev;
	std::mt19937 rng(dev());

	for (int i = 0; i < N; ++i) {
		std::uniform_int_distribution<std::mt19937::result_type> distribution(0,N - 1);
		randomValues[i] = distribution(rng);
	}

	return randomValues;
}

vector<size_t> NeuralNetwork::randomisedIndexes(const int n) {
	vector<size_t> randomValues(n);
	std::random_device dev;
	std::mt19937 rng(dev());

	for (int i = 0; i < n; ++i) {
		std::uniform_int_distribution<std::mt19937::result_type> distribution(0,n - 1);
		randomValues[i] = distribution(rng);
	}

	return randomValues;
}

template <typename T>
vector<T> NeuralNetwork::randomiseVector(const vector<T>& values, const vector<size_t>& randomIndices) {
	const size_t n = randomIndices.size();
	vector<T> randomisedValues(n);

	for (int i = 0; i < n; ++i) {
		randomisedValues[i] = values[randomIndices[i]];
	}

	return randomisedValues;
}

void NeuralNetwork::setLayerValues(const std::string& filePath) {
	DataStorage::retrieveData(*this, filePath);
}

std::vector<Layer>& NeuralNetwork::getLayers() {return layers;}


void NeuralNetwork::threadExample(bool& stopped) {
	std::string _;
	std::getline(std::cin, _);
	std::cout << "Process Terminated" << std::endl;

	stopped = true;
}