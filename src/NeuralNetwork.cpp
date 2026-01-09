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

NeuralNetwork::NeuralNetwork(const vector<vector<double>>& data, const vector<int>& labels, vector<int> shape, const double learningRate, const int batchSize)
	: size(shape.size() - 1), learningRate(learningRate), data(data), labels(labels), N(labels.size()), batchSize(batchSize)
{
	for (size_t i = 0; i < size; i++) {
		layers.emplace_back(shape[i], shape[i + 1]); // constructs Layer object within the push
	}
}

void NeuralNetwork::train(const double trainingSplit) {
	loadData(trainingSplit);

	clock_t beginTime = -30000;
	double previousCost = 1000;
	bool stopped = false;

	// Threads operation
	// std::thread terminateThread([this, &stopped] {
	// 		threadExample(stopped);
	// 	});

	while (!stopped) {
		// Train neural network
		for (int i = 1; i < trainingData.size() / batchSize; ++i) {
			const size_t startPoint = batchSize * (i - 1);
			size_t endPoint = batchSize * i;

			if (endPoint > trainingData.size())
				endPoint = trainingData.size();

			const vector<vector<double>> trainingBatch = Numpty::deepCopy2D(trainingData, startPoint, endPoint);
			const vector<int> labelsBatch = Numpty::copy(trainingLabels, startPoint, endPoint);

			auto batchOutput = forwardPass(trainingBatch);
			// const double lossValue = loss(batchOutput, labelsBatch);
			backpropagation(trainingBatch, labelsBatch);
			applyGradients();
			// throw std::runtime_error("Batch complete");
			// std::cout << lossValue << '\n';
		}

		if (clock() - beginTime > 3000 || iterations % 1000 == 0) {
			beginTime = clock();
			const double costValue = loss(forwardPass(trainingData), trainingLabels);

			// if (previousCost - costValue < 0.0001) {
			// 	stopped = true;
			// }

			previousCost = costValue;
			std::cout << costValue << std::endl;

			const double accuracy = cost(forwardPass(testingData), testingLabels);
			std::cout << accuracy << "/" << testingData.size() << '\n' << '\n';
		}
		iterations++;
	}
	std::cout << loss(trainingData, trainingLabels) << std::endl;
	DataStorage::saveData(*this, "");

	// if (terminateThread.joinable()) {
	// 	terminateThread.join();
	// }
}

vector<vector<double>> NeuralNetwork::forwardPass(const vector<vector<double> > &inputs) {
	vector<vector<double>> batchOutput;
	vector<double> outputs(inputs[0].size());
	for (const auto & input : inputs) {
		outputs = layers[0].calculateLayerOutput(input, "Tanh");
		for (size_t j = 1; j < size - 1; j++) {
			outputs = layers[j].calculateLayerOutput(outputs, "Tanh");
		}
		batchOutput.push_back(layers[size - 1].calculateLayerOutput(outputs, "Softmax"));
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
	vector<double> networkErrors(size - 1);
	const vector<int> correctedLabels = Numpty::subtractScalar(trainingLabels, 1);
	const int batchSize = trainingData.size();

	// output layer //
	Layer& currentLayer = layers[size - 1];

	currentLayer.batchErrorSignals.clear();
	currentLayer.batchErrorSignals.reserve(batchSize);
	for (int j = 0; j < batchSize; ++j) {
		vector<double> errorSignal = Numpty::hotVectorOutput(currentLayer.batchInputs[j], correctedLabels[j], currentLayer.numNodes);

		for (int numNodes = 0; numNodes < currentLayer.numNodes; ++numNodes) {
			for (int nodesIn = 0; nodesIn < currentLayer.nodesIn; ++nodesIn) {
				currentLayer.costGradientWeights[numNodes][nodesIn] += errorSignal[numNodes] * currentLayer.batchInputs[j][nodesIn];
			}
			currentLayer.costGradientBiases[numNodes] += errorSignal[numNodes];
		}

		currentLayer.batchErrorSignals.push_back(errorSignal);
	}
	Numpty::multiplyByScalar(currentLayer.costGradientWeights, 1.0/batchSize);
	Numpty::multiplyByScalar(currentLayer.costGradientBiases, 1.0/batchSize);

	// hidden layer //
	Layer& hiddenCurrentLayer = layers[size - 2];
	const Layer& previousLayer = layers[size - 1];

	const vector<vector<double>> transposedWeightMatrix = Numpty::transpose(hiddenCurrentLayer.weights);
	for (int j = 0; j < batchSize; ++j) {
		vector<double> hiddenErrors = Numpty::hiddenErrors(transposedWeightMatrix, previousLayer.batchErrorSignals[j]);
		for (int i = 0; i < hiddenErrors.size(); ++i) {
			hiddenErrors[i] *= 1 - std::pow(previousLayer.batchInputs[j][i], 2);
		}

		for (int numNodes = 0; numNodes < hiddenCurrentLayer.numNodes; ++numNodes) {
			for (int nodesIn = 0; nodesIn < hiddenCurrentLayer.nodesIn; ++nodesIn) {
				hiddenCurrentLayer.costGradientWeights[numNodes][nodesIn] += hiddenErrors[numNodes] * hiddenCurrentLayer.batchInputs[j][nodesIn];
			}
			hiddenCurrentLayer.costGradientBiases[numNodes] += hiddenErrors[numNodes];
		}
	}
	Numpty::multiplyByScalar(hiddenCurrentLayer.costGradientWeights, 1.0/batchSize);
	Numpty::multiplyByScalar(hiddenCurrentLayer.costGradientBiases, 1.0/batchSize);
}

void NeuralNetwork::applyGradients() {
	for (Layer & layer : layers) {
		layer.applyGradients(learningRate);
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

void NeuralNetwork::setLayerValues(const std::string& filePath) {
	DataStorage::retrieveData(*this, filePath);
}

void NeuralNetwork::printLayerValues() {
	int i = 1;
	for (Layer & layer : layers) {
		std::cout << "Applying gradients to layer: " << i << '\n';
		NumptyHelper::print2D(layer.weights);
		NumptyHelper::print2D(layer.costGradientWeights);
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