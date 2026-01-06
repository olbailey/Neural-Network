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
	int batchSize;

	std::vector<std::vector<double>> trainingData;
	std::vector<int> trainingLabels;
	std::vector<std::vector<double>> testingData;
	std::vector<int> testingLabels;

public:
	NeuralNetwork(const std::vector<std::vector<double>>& data, const std::vector<int>& labels, std::vector<int> shape, double learningRate, int batchSize);

	void train(double trainingSplit);

	std::vector<Layer>& getLayers();

	void setLayerValues(const std::string &filePath);

	static std::vector<size_t> randomisedIndexes(int n) ;

private:

	void forwardPass(const std::vector<std::vector<double>> &inputs);

	/**
	 * Implementation of categorical cross-entropy loss
	 * @param predicts
	 * @param expectedOutputs
	 * @return
	 */
	static double loss(const std::vector<std::vector<double>>& predicts, const std::vector<int>& expectedOutputs);

	static double cost(const std::vector<std::vector<double> > &predicts, const std::vector<int> &expectedOutputs) ;

	void backpropagation(const std::vector<std::vector<double>> &trainingData, const std::vector<int> &trainingLabels);

	void applyGradients();

	void loadData(double trainingSplit);

	std::vector<size_t> randomisedIndexes() const;

	template<class T>
	std::vector<T> randomiseVector(const std::vector<T>& values, const std::vector<size_t>& randomIndices);

	static void threadExample(bool& stopped);
};

#endif // !NEURALNETWORK_H
