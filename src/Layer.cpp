#include "Layer.h"

#include <vector>
#include <cmath>
#include <string>

#include "Numpty.h"
#include "NumptyHelper.h"

using std::vector;


Layer::Layer(const size_t inputN, const size_t currentN) {
	nodesIn = inputN;
	numNodes = currentN;

	costGradientWeights = Numpty::zeros(currentN, inputN);
	costGradientBiases = Numpty::zeros(currentN);
	weights = Numpty::random(currentN, inputN);
	biases = Numpty::zeros(currentN);
}

vector<double> Layer::calculateLayerOutput(const vector<double>& inputs, const std::string& activationFunctionName) const {
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

	return layerOutputs;
}

void Layer::applyGradients(const double learningRate) {
	for (size_t i = 0; i < numNodes; i++) {
		biases[i] -= costGradientBiases[i] * learningRate;

		for (int j = 0; j < nodesIn; j++) {
			weights[i][j] -= costGradientWeights[i][j] * learningRate;
		}
	}
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
	for (double & input : inputs) {
		input = exp(input);
	}

	const double normBase = Numpty::sum(inputs);

	for (double & input : inputs) {
		input = input / normBase;
	}
}

std::string Layer::getShape() const {
	return '(' + std::to_string(nodesIn) + ',' + std::to_string(numNodes) + ')';
}

std::vector<std::vector<double> > Layer::getWeights() const {return weights;}

std::vector<double> Layer::getBiases() const {return biases;}

void Layer::setWeights(const vector<vector<double>>& newWeights) {weights = newWeights;}

void Layer::setBiases(const vector<double>& newBiases) {biases = newBiases;}