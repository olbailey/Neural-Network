#pragma once
#ifndef NUMPTY_H
#define NUMPTY_H

#include <vector>

namespace Numpty {
	/* Generates a matrix holding double values 
	that are random and normally distributed 
	with a mean of 0 and a standard deviation of 1
	\param nodesOut Number of rows
	\param nodeIn Number of columns
	\return A matrix holding double values 
	that are random and normally distributed*/
	std::vector<std::vector<double>> random(size_t nodesOut, size_t nodesIn);

	std::vector<double> zeros(size_t size);

	std::vector<std::vector<double>> zeros(size_t rows, size_t height);

	std::vector<int> subtractScalar(std::vector<int> a, int b);

	void addScalar(std::vector<int>& a, int b);

	void multiplyByScalar(std::vector<double>& a, double b);

	double sum(const std::vector<double>& values);

	std::vector<int> argmax(const std::vector<std::vector<double>> &values);

	double dot(std::vector<double> a, std::vector<double> b);

	double mean(const std::vector<double>& values);

	int equal(const std::vector<int> &a, const std::vector<int> &b);

	std::vector<double> logarithm(const std::vector<std::vector<double>> &inputs, const std::vector<int> &);

	std::vector<std::vector<double>>
		deepCopy2D(const std::vector<std::vector<double>> &input, size_t l, size_t r);

	std::vector<int> copy(std::vector<int> input, size_t l, size_t r);
};

#endif