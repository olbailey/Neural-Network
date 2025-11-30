#include "Layer.h"

#include <vector>
#include <cmath>
#include <string>

#include "Numpty.h"
#include "NumptyHelper.h"

using std::vector;


Layer::Layer(const size_t inputN, const size_t outputN) {
	nodesIn = inputN;
	nodesOut = outputN;

	costGradientWeights = Numpty::zeros(outputN, inputN);
	costGradientBiases = Numpty::zeros(outputN);
	weights = Numpty::random(outputN, inputN);
	biases = Numpty::zeros(outputN);
}

vector<double> Layer::calculateLayerOutput(const vector<double>& inputs, const std::string& activationFunctionName) const {
	vector<double> layerOutputs(nodesOut);

	for (size_t i = 0; i < nodesOut; i++) {
		layerOutputs[i] = Numpty::dot(inputs, weights[i]) + biases[i];
	}

	if (activationFunctionName == "Sigmoid")
		sigmoid(layerOutputs);
	else if (activationFunctionName == "Softmax")
		softmax(layerOutputs);

	return layerOutputs;
}

void Layer::applyGradients(const double learningRate) {
	for (size_t i = 0; i < weights.size(); i++) {
		biases[i] -= costGradientBiases[i] * learningRate;

		for (int j = 0; j < weights[i].size(); j++) {
			weights[i][j] -= costGradientWeights[i][j] * learningRate;
		}
	}
}

void Layer::sigmoid(vector<double>& inputs) {
	for (size_t i = 0; i < inputs.size(); i++) {
		inputs[i] = 1 / (1 + exp(-inputs[i]));
	}
}

void Layer::softmax(vector<double>& inputs) {
	for (size_t i = 0; i < inputs.size(); i++) {
		inputs[i] = exp(inputs[i]);
	}

	const double normBase = Numpty::sum(inputs);

	for (size_t i = 0; i < inputs.size(); i++) {
		inputs[i] = inputs[i] / normBase;
	}
}

std::string Layer::getShape() const {
	return '(' + std::to_string(nodesIn) + ',' + std::to_string(nodesOut) + ')';
}

std::vector<std::vector<double> > Layer::getWeights() const {return weights;}

std::vector<double> Layer::getBiases() const {return biases;}

void Layer::setWeights(const vector<vector<double>>& newWeights) {weights = newWeights;}

void Layer::setBiases(const vector<double>& newBiases) {biases = newBiases;}