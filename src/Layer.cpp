#include "Layer.h"

#include <algorithm>
#include <vector>
#include <cmath>
#include <iostream>
#include <string>

#include "Numpty.h"

using std::vector;
using std::exp;

Layer::Layer(const size_t inputN, const size_t currentN) {
	nodesIn = inputN;
	numNodes = currentN;

	costGradientWeights = Numpty::zeros(currentN, inputN);
	costGradientBiases = Numpty::zeros(currentN);
	weights = Numpty::xavier(currentN, inputN);
	biases = Numpty::zeros(currentN);
}

vector<double> Layer::calculateLayerOutput(const vector<double>& inputs, const std::string& activationFunctionName) {
	vector<double> layerOutputs(numNodes);

	for (size_t i = 0; i < numNodes; i++) {
		layerOutputs[i] = Numpty::dot(inputs, weights[i]) + biases[i];
	}

	if (activationFunctionName == "Sigmoid")
		sigmoid(layerOutputs);
	else if (activationFunctionName == "Tanh")
		hyperbolicTangent(layerOutputs);
	else if (activationFunctionName == "Softmax")
		softmax(layerOutputs);

	batchOutputs.push_back(layerOutputs);
	return layerOutputs;
}

void Layer::applyGradients(const double learningRate) {
	for (size_t i = 0; i < numNodes; i++) {
		biases[i] -= costGradientBiases[i] * learningRate;

		for (int j = 0; j < nodesIn; j++) {
			weights[i][j] -= costGradientWeights[i][j] * learningRate;
		}
	}
	Numpty::resetMatrixToZero(costGradientWeights);
	Numpty::resetVectorToZero(costGradientBiases);
}

void Layer::backpropagationOutputLayer
			(Layer &currentLayer, const Layer &previousLayer, const vector<int> &trainingLabels) {
	const int batchSize = trainingLabels.size();
	const vector<int> correctedLabels = Numpty::subtractScalar(trainingLabels, 1);

	currentLayer.batchErrorSignals.clear();
	currentLayer.batchErrorSignals.reserve(batchSize);

	for (int j = 0; j < batchSize; ++j) {
		vector<double> errorSignal =
			Numpty::hotVectorOutput(currentLayer.batchOutputs[j], correctedLabels[j], currentLayer.numNodes);

		for (int numNodes = 0; numNodes < currentLayer.numNodes; ++numNodes) {
			for (int nodesIn = 0; nodesIn < currentLayer.nodesIn; ++nodesIn) {
				currentLayer.costGradientWeights[numNodes][nodesIn]
					+= errorSignal[numNodes] * previousLayer.batchOutputs[j][nodesIn];
			}
			currentLayer.costGradientBiases[numNodes] += errorSignal[numNodes];
		}

		currentLayer.batchErrorSignals.push_back(errorSignal);
	}

	Numpty::multiplyByScalar(currentLayer.costGradientWeights, 1.0/batchSize);
	Numpty::multiplyByScalar(currentLayer.costGradientBiases, 1.0/batchSize);
	currentLayer.batchOutputs.clear();
}

void Layer::backpropagationHiddenLayer
		(Layer &currentLayer, const Layer &previousLayer, const std::vector<vector<double>> &trainingData) {
	const int batchSize = trainingData.size();
	const vector<vector<double>> transposedWeightMatrix = Numpty::transpose(previousLayer.weights);

	for (int j = 0; j < batchSize; ++j) {
		vector<double> hiddenErrors = Numpty::hiddenErrors(transposedWeightMatrix, previousLayer.batchErrorSignals[j]);
		for (int i = 0; i < hiddenErrors.size(); ++i) {
			hiddenErrors[i] *= 1 - std::pow(currentLayer.batchOutputs[j][i], 2);
		}

		for (int numNodes = 0; numNodes < currentLayer.numNodes; ++numNodes) {
			for (int nodesIn = 0; nodesIn < currentLayer.nodesIn; ++nodesIn) {
				currentLayer.costGradientWeights[numNodes][nodesIn] += hiddenErrors[numNodes] * trainingData[j][nodesIn];
			}
			currentLayer.costGradientBiases[numNodes] += hiddenErrors[numNodes];
		}

		currentLayer.batchErrorSignals.push_back(hiddenErrors);
	}

	Numpty::multiplyByScalar(currentLayer.costGradientWeights, 1.0/batchSize);
	Numpty::multiplyByScalar(currentLayer.costGradientBiases, 1.0/batchSize);
	currentLayer.batchOutputs.clear();
}


void Layer::sigmoid(vector<double>& inputs) {
	for (double & input : inputs) {
		input = 1 / (1 + exp(-input));
	}
}

void Layer::hyperbolicTangent(vector<double>& inputs) {
	for (double & input : inputs) {
		input = std::tanh(input);
	}
}

void Layer::softmax(vector<double>& inputs) {
	const double maxVal = *std::ranges::max_element(inputs);

	for (double & input : inputs) {
		input = exp(input - maxVal);
	}

	const double normBase = Numpty::sum(inputs);

	for (double & input : inputs) {
		input = input / normBase;
	}

	if (Numpty::sum(inputs) < 0.99) {
		std::cout << Numpty::sum(inputs);
		throw std::runtime_error("softmax doesn't sum to 1");
	}
}

std::string Layer::getShape() const {
	return '(' + std::to_string(nodesIn) + ',' + std::to_string(numNodes) + ')';
}

std::vector<std::vector<double> > Layer::getWeights() const {return weights;}

std::vector<double> Layer::getBiases() const {return biases;}

void Layer::setWeights(const vector<vector<double>>& newWeights) {weights = newWeights;}

void Layer::setBiases(const vector<double>& newBiases) {biases = newBiases;}

void Layer::checkInvalidNum(const double x) {
	if (std::isnan(x)) throw std::runtime_error("NaN detected!");
	if (std::isinf(x)) throw std::runtime_error("Inf detected!");
}