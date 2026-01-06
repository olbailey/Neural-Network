#include "Layer.h"

#include <algorithm>
#include <vector>
#include <cmath>
#include <string>

#include "Numpty.h"
#include "NumptyHelper.h"

using std::vector;
using std::exp;

Layer::Layer(const size_t inputN, const size_t currentN) {
	nodesIn = inputN;
	numNodes = currentN;

	costGradientWeights = Numpty::zeros(currentN, inputN);
	costGradientBiases = Numpty::zeros(currentN);
	weights = Numpty::random(currentN, inputN);
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
}

void Layer::sigmoid(vector<double>& inputs) {
	for (double & input : inputs) {
		input = 1 / (1 + exp(-input));
	}
}

void Layer::hyperbolicTangent(vector<double>& inputs) {
	for (double & input : inputs) {
		checkInvalidNum(input);
		if (input > 20)
			input = 1.0;
		else if (input < -20)
			input = -1.0;
		else {
			const double e2x = exp(2 * input);
			input = (e2x - 1) / (e2x + 1);
		}
		checkInvalidNum(input);

		/* This is equivalent to (e^x - e^-x) / (e^x + e^-x)
		 * because multiply fraction by e^x/e^x,
		 * and it simplifies to (e^2x - 1) / (e^2x + 1)
		 */
	}
}

void Layer::softmax(vector<double>& inputs) {
	const double maxVal = *std::ranges::max_element(inputs);

	for (double & input : inputs) {
		checkInvalidNum(input);
		input = exp(input - maxVal);
		checkInvalidNum(input);
	}

	const double normBase = Numpty::sum(inputs);

	for (double & input : inputs) {
		input = input / normBase;
		checkInvalidNum(input);
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