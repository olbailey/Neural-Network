#pragma once

#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <string>

class Layer {
public:
	size_t nodesIn;
	size_t numNodes;
	std::vector<std::vector<double>> costGradientWeights;
	std::vector<double> costGradientBiases;
	std::vector<std::vector<double>> weights;
	std::vector<double> biases;

	Layer(size_t inputN, size_t currentN);

	/**
	 * Forward pass of layer
	 * @param inputs The values
	 * @param activationFunctionName Name of activation function to be used (Sigmoid, Tanh, Softmax)
	 * @return Vector values with activation function applied
	 */
	std::vector<double> calculateLayerOutput(const std::vector<double>& inputs, const std::string& activationFunctionName) const;

	void applyGradients(double learningRate);

	std::string getShape() const;

	std::vector<std::vector<double>> getWeights() const;

	std::vector<double> getBiases() const;

	void setWeights(const std::vector<std::vector<double>>& newWeights);

	void setBiases(const std::vector<double>& newBiases);

private:
	static void sigmoid(std::vector<double> &inputs);

	// Tanh
	static void hyperbolicTangent(std::vector<double> &inputs);

	static void softmax(std::vector<double> &inputs);

};

#endif // !LAYER_H