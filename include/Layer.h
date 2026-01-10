#pragma once

#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <string>

class Layer {
public:
	size_t nodesIn;
	size_t numNodes;

	// Basic values
	std::vector<std::vector<double>> weights;
	std::vector<double> biases;

	// cost gradient storage
	std::vector<std::vector<double>> costGradientWeights;
	std::vector<double> costGradientBiases;

	// backpropagation information
	std::vector<std::vector<double>> batchOutputs;
	std::vector<std::vector<double>> batchErrorSignals;

	// Momentum implementation
	std::vector<std::vector<double>> velocityWeights;
	std::vector<double> velocityBiases;

	Layer(size_t inputN, size_t currentN);

	/**
	 * Forward pass of layer
	 * @param inputs The values
	 * @param activationFunctionName Name of activation function to be used (Sigmoid, Tanh, Softmax)
	 * @param simulation
	 * @return Vector values with activation function applied
	 */
	std::vector<double> calculateLayerOutput(const std::vector<double> &inputs, const std::string &activationFunctionName, bool simulation);

	void applyGradients(double learningRate, double beta);

	/**
	 * Used only for backpropagation of output layer. must be done before any other backpropagation
	 * @param currentLayer Output layer
	 * @param previousLayer Final hidden layer
	 * @param trainingLabels Original batch training labels
	 * @param batchSize
	 */
	static void backpropagationOutputLayer
			(Layer &currentLayer, const Layer &previousLayer, const std::vector<int> &trainingLabels);

	/**
	 * Used only for hidden layer. must be done after its previous layer has been done
	 * @param currentLayer Current hidden layer
	 * @param previousLayer Layer that this layers outputs go to
	 * @param trainingData original training data for batch
	 * @param batchSize
	 */
	static void backpropagationHiddenLayer
			(Layer &currentLayer, const Layer &previousLayer, const std::vector<std::vector<double>> &trainingData);

	std::string getShape() const;

	std::vector<std::vector<double>> getWeights() const;

	std::vector<double> getBiases() const;

	void setWeights(const std::vector<std::vector<double>>& newWeights);

	void setBiases(const std::vector<double>& newBiases);

	static void checkInvalidNum(double x);

private:
	static void sigmoid(std::vector<double> &inputs);

	// Tanh
	static void hyperbolicTangent(std::vector<double> &inputs);

	static void softmax(std::vector<double> &inputs);

};

#endif // !LAYER_H