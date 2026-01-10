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

NeuralNetwork::NeuralNetwork(const vector<vector<double>>& data, const vector<int>& labels,
							vector<int> shape, const double learningRate, const int batchSize)
	: size(shape.size() - 1), learningRate(learningRate), data(data),
		labels(labels), N(labels.size()), batchSize(batchSize)
{
	for (size_t i = 0; i < size; i++) {
		layers.emplace_back(shape[i], shape[i + 1]); // constructs Layer object within the push
	}
}

void NeuralNetwork::train(const double trainingSplit) {
	loadData(trainingSplit);

	bool stopped = false;

	// Threads operation
	std::thread terminateThread([this, &stopped] {
			threadExample(stopped);
		});

	printNetworkPerformance();

	while (!stopped) {
		// Train neural network
		for (int i = 1; i < trainingData.size() / batchSize; ++i) {
			const size_t startPoint = batchSize * (i - 1);
			size_t endPoint = batchSize * i;

			if (endPoint > trainingData.size())
				endPoint = trainingData.size();

			const vector<vector<double>> trainingBatch = Numpty::deepCopy2D(trainingData, startPoint, endPoint);
			const vector<int> labelsBatch = Numpty::copy(trainingLabels, startPoint, endPoint);

			auto batchOutput = forwardPass(trainingBatch, false);
			backpropagation(trainingBatch, labelsBatch);
			applyGradients();
		}

		iterations++;

		if (iterations % 1000 == 0)
			printNetworkPerformance();
	}
	std::cout << loss(trainingData, trainingLabels) << std::endl;
	DataStorage::saveData(*this, "");

	if (terminateThread.joinable()) {
		terminateThread.join();
	}
}

vector<vector<double>> NeuralNetwork::forwardPass(const vector<vector<double> > &inputs, const bool simulation) {
	vector<vector<double>> batchOutput;
	vector<double> outputs(inputs[0].size());
	for (const auto & input : inputs) {
		outputs = layers[0].calculateLayerOutput(input, "Tanh", simulation);
		for (size_t j = 1; j < size - 1; j++) {
			outputs = layers[j].calculateLayerOutput(outputs, "Tanh", simulation);
		}
		batchOutput.push_back(layers[size - 1].calculateLayerOutput(outputs, "Softmax", simulation));
	}
	return batchOutput;
}

double NeuralNetwork::loss(const vector<vector<double>>& predicts, const vector<int>& expectedOutputs) {
	const vector<int> correctedLabels = Numpty::subtractScalar(expectedOutputs, 1);

	vector<double> losses = Numpty::logarithm(predicts, correctedLabels);
	Layer::checkInvalidNum(losses[0]);

	Numpty::multiplyByScalar(losses, -1);

	return Numpty::mean(losses);
}

void NeuralNetwork::backpropagation(const vector<vector<double>>& trainingData, const vector<int> &trainingLabels) {
	Layer::backpropagationOutputLayer
		(layers[size - 1], layers[size - 2], trainingLabels);

	for (int i = size - 2; i > 0; --i) {
		Layer::backpropagationHiddenLayer(layers[i], layers[i + 1], layers[i - 1].batchOutputs);
	}

	Layer::backpropagationHiddenLayer(layers[0], layers[1], trainingData);
}

void NeuralNetwork::applyGradients() {
	for (Layer & layer : layers) {
		layer.applyGradients(learningRate, beta);
	}
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

void NeuralNetwork::printNetworkPerformance() {
	const auto testingPredicts = forwardPass(testingData, true);

	std::cout << "Iterations: " << iterations << '\n';
	// std::cout << "Training Data Loss: " << loss(forwardPass(trainingData), trainingLabels) << std::endl;
	std::cout << "Testing Data Loss: " << loss(testingPredicts, testingLabels) << std::endl;

	std::cout << "Accuracy: " << cost(testingPredicts, testingLabels) << "/" << testingData.size() << '\n' << '\n';
}

void NeuralNetwork::setLayerValues(const std::string& filePath) {
	DataStorage::retrieveData(*this, filePath);
}

void NeuralNetwork::printLayerValues() {
	int i = 1;
	for (Layer & layer : layers) {
		std::cout << "Applying gradients to layer: " << i << '\n';
		std::cout << "Weights:\n";
		NumptyHelper::print2D(layer.weights);
		NumptyHelper::print2D(layer.costGradientWeights);
		std::cout << "biases:\n";
		NumptyHelper::print1D(layer.biases);
		NumptyHelper::print1D(layer.costGradientBiases);
		i++;
	}
}

std::vector<Layer>& NeuralNetwork::getLayers() {return layers;}


void NeuralNetwork::threadExample(bool& stopped) {
	std::string _;
	std::getline(std::cin, _);
	std::cout << "Process Terminated" << std::endl;

	stopped = true;
}