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

using std::vector;

NeuralNetwork::NeuralNetwork(vector<int> shape, const double learningRate)
	: size(shape.size() - 1),
	learningRate(learningRate) 
{
	for (size_t i = 0; i < size; i++) {
		layers.emplace_back(shape[i], shape[i + 1]); // constructs Layer object within the push
		// given it knows that you are pushing a layer object to it as that is what the vector stores
	}
}

void NeuralNetwork::train(const vector<vector<double>>& data, const vector<int>& expectedOutputs, const int batchSize, const double trainingSplit) {
	std::cout << std::thread::hardware_concurrency() << std::endl;
	const vector<vector<double>> randomData = randomiseData(data);

	const size_t numDataPoints = randomData.size();

	const long testingCutOff = static_cast<long>(std::floor(numDataPoints * trainingSplit));

	const vector<vector<double>> trainingData = Numpty::deepCopy2D(randomData, 0, testingCutOff);
	const vector<int> trainingExpectedOutputs = Numpty::copy(expectedOutputs, 0, testingCutOff);

	const vector<vector<double>> testingData = Numpty::deepCopy2D(randomData, testingCutOff, numDataPoints);
	const vector<int> testingExpectedOutputs = Numpty::copy(expectedOutputs, testingCutOff, numDataPoints);

	clock_t beginTime = -30000;
	double previousCost = 1000;
	int count = 0;
	bool stopped = false;

	// Threads operation
	std::thread terminateThread(&NeuralNetwork::threadExample, this, std::ref(stopped));

	while (!stopped) {
		// Train neural network
		for (int i = 1; i < numDataPoints / batchSize; i++) {
			const size_t startPoint = batchSize * (i - 1);
			size_t endPoint = batchSize * i;

			if (endPoint > trainingData.size())
				endPoint = trainingData.size();

			const vector<vector<double>> trainingBatch = Numpty::deepCopy2D(trainingData, startPoint, endPoint);
			const vector<int> expectedBatch = Numpty::copy(trainingExpectedOutputs, startPoint, endPoint);

			learn(trainingBatch, expectedBatch);
		}

		if (clock() - beginTime > 30000 || count % 1000 == 0) {
			beginTime = clock();
			const double costValue = loss(trainingData, trainingExpectedOutputs);

			if (previousCost - costValue < 0.0001) {
				stopped = true;
			}

			previousCost = costValue;
			std::cout << costValue << std::endl;

			const double accuracy = cost(testingData, testingExpectedOutputs);
			std::cout << accuracy << "/" << testingData.size() << '\n';
		}
		count++;
	}
	std::cout << loss(trainingData, trainingExpectedOutputs) << std::endl;
	DataStorage::saveData(*this, "");

	if (terminateThread.joinable()) {
		terminateThread.join();
	}
}

vector<double> NeuralNetwork::calculateOutputs(vector<double> inputs) const {
	for (size_t i = 0; i < size - 1; i++) {
		inputs = layers[i].calculateLayerOutput(inputs, "Sigmoid");
	}
	inputs = layers[size - 1].calculateLayerOutput(inputs, "Softmax");

	return inputs;
}

vector<vector<double>> NeuralNetwork::classify(const vector<vector<double>> &inputs) const {
	vector outputs(inputs.size(), vector<double>(2));

	for (size_t i = 0; i < inputs.size(); i++) {
		outputs[i] = calculateOutputs(inputs[i]);
	}

	return outputs;
}

double NeuralNetwork::loss(const vector<vector<double>>& trainingData, const vector<int>& expectedOutputs) const {
	const vector<vector<double>> predicts = classify(trainingData);
	const vector<int> expectedCopy = Numpty::copy(expectedOutputs, 0, expectedOutputs.size());
	const vector<int> correctedExpectedOutputs = Numpty::subtractScalar(expectedCopy, 1);

	vector<double> losses = Numpty::logarithm(predicts, correctedExpectedOutputs);

	Numpty::multiplyByScalar(losses, -1);

	return Numpty::mean(losses);
}

double NeuralNetwork::cost(const vector<vector<double> > &testData, const vector<int> &expectedOutputs) const {
	const vector<vector<double>> predicts = classify(testData);
	vector<int> predictedOutputChoices = Numpty::argmax(predicts);
	Numpty::addScalar(predictedOutputChoices, 1);

	return Numpty::equal(predictedOutputChoices, expectedOutputs);
}

void NeuralNetwork::learn(const vector<vector<double>>& trainingData, const vector<int> &expectedOutputs) {
	const double originalCost = loss(trainingData, expectedOutputs);
	constexpr double H = 0.0001;

	for (Layer& layer : layers) {
		double deltaCost;

		for (size_t nodeIn = 0; nodeIn < layer.nodesOut; nodeIn++) {
			for (size_t nodeOut = 0; nodeOut < layer.nodesIn; nodeOut++) {
				layer.weights[nodeIn][nodeOut] = layer.weights[nodeIn][nodeOut] + H;
				deltaCost = loss(trainingData, expectedOutputs) - originalCost;
				layer.weights[nodeIn][nodeOut] = layer.weights[nodeIn][nodeOut] - H;
				layer.costGradientWeights[nodeIn][nodeOut] = deltaCost / H;
			}
		}

		for (size_t biasesIndex = 0; biasesIndex < layer.nodesOut; biasesIndex++) {
			layer.biases[biasesIndex] = layer.biases[biasesIndex] + H;
			deltaCost = loss(trainingData, expectedOutputs) - originalCost;
			layer.biases[biasesIndex] = layer.biases[biasesIndex] - H;
			layer.costGradientBiases[biasesIndex] = deltaCost / H;
		}

		layer.applyGradients(learningRate);
	}
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

vector<vector<double>> NeuralNetwork::randomiseData(vector<vector<double>> dataCopy) {
	vector<vector<double>> dataRandomised;
	dataRandomised.reserve(dataCopy.size());
	std::random_device dev;
	std::mt19937 rng(dev());

	for (int i = 0; i < dataCopy.size(); ++i) {
		std::uniform_int_distribution<std::mt19937::result_type> distribution(0,dataCopy.size()-1);
		const size_t randIndex = distribution(rng);
		dataRandomised.push_back(dataCopy[randIndex]);
		dataCopy.erase(dataCopy.begin()+randIndex);
	}

	return dataRandomised;
}