#pragma once
#ifndef NUMPTY_H
#define NUMPTY_H

#include <random>
#include <vector>

namespace Numpty {
	inline std::mt19937 engine{ std::random_device{}() };

	/** Generates a matrix holding double values
	* that are random and normally distributed
	* with a mean of 0 and a standard deviation of 1
	* @param nodesOut Number of rows
	* @param nodesIn Number of columns
	* @return A matrix holding double values that are random and normally distributed
	*/
	std::vector<std::vector<double>> random(size_t nodesOut, size_t nodesIn);

	/**
	 * Xavier initialization chooses weight scales so that information
	 * neither explodes nor vanishes as it flows forward or backward
	 * through the network.
	 * Weights sampled uniformly around 0 between limits based on,
	 * number of inputs and outputs from a neuron.
	 * @param nodesOut
	 * @param nodesIn
	 * @return
	 */
	std::vector<std::vector<double>> xavier(size_t nodesOut, size_t nodesIn);

	std::vector<double> zeros(size_t size);

	std::vector<std::vector<double>> zeros(size_t rows, size_t height);

	std::vector<int> subtractScalar(std::vector<int> a, int b);

	void addScalar(std::vector<int>& a, int b);

	void multiplyByScalar(std::vector<double>& a, double b);

	void multiplyByScalar(std::vector<std::vector<double>>& a, double b);

	void resetMatrixToZero(std::vector<std::vector<double>> a);

	void resetVectorToZero(std::vector<double> a);

	std::vector<std::vector<double>>
		combineMatrices(const std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b);

	std::vector<double> combineVectors(const std::vector<double> &a, const std::vector<double> &b);

	double sum(const std::vector<double>& values);

	std::vector<int> argmax(const std::vector<std::vector<double>> &values);

	double dot(std::vector<double> a, std::vector<double> b);

	double mean(const std::vector<double>& values);

	int equal(const std::vector<int> &a, const std::vector<int> &b);

	std::vector<double> logarithm(const std::vector<std::vector<double>> &inputs, const std::vector<int> &);

	std::vector<double> hotVectorOutput(const std::vector<double> &outputs, int label, int classesNum);

	std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> a);

	std::vector<double>
		hiddenErrors(const std::vector<std::vector<double>> &transposedWeights, const std::vector<double> &errorSignal);


	std::vector<std::vector<double>>
		deepCopy2D(const std::vector<std::vector<double>> &input, size_t l, size_t r);

	std::vector<int> copy(const std::vector<int> &input, size_t l, size_t r);
};

#endif