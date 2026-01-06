#pragma once

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Layer.h"
#include <vector>
#include <string>

class NeuralNetwork {
	std::vector<Layer> layers;
	size_t size;
	double learningRate;
	size_t iterations = 0;

	const std::vector<std::vector<double>>& data;
	const std::vector<int>& labels;
	const size_t N;

	std::vector<std::vector<double>> trainingData;
	std::vector<int> trainingLabels;
	std::vector<std::vector<double>> testingData;
	std::vector<int> testingLabels;

public:
	NeuralNetwork(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, std::vector<int> shape, double learningRate);

	void train(int batchSize, double trainingSplit);

	std::vector<Layer>& getLayers();

	void setLayerValues(const std::string &filePath);

private:

	std::vector<std::vector<double>> forwardPass(const std::vector<std::vector<double>> &inputs) const;

	/**
	 * Implementation of categorical cross-entropy loss
	 * @param predicts
	 * @param expectedOutputs
	 * @return
	 */
	static double loss(const std::vector<std::vector<double>>& predicts, const std::vector<int>& expectedOutputs);

	double cost(const std::vector<std::vector<double> > &testData, const std::vector<int> &expectedOutputs) const;

	void gradientDescent(const std::vector<std::vector<double>> &trainingData, const std::vector<int> &expectedOutputs, double originalCost);

	void backpropagation(const std::vector<std::vector<double>> &trainingData, const std::vector<int> &expectedOutputs);

	void loadData(double trainingSplit);

	std::vector<size_t> randomisedIndexes() const;

	template<class T>
	std::vector<T> randomiseVector(const std::vector<T>& values, const std::vector<size_t>& randomIndices);

	static void threadExample(bool& stopped);
};

#endif // !NEURALNETWORK_H
