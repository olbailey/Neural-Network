#include "Numpty.h"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "NumptyHelper.h"


using std::vector;

vector<vector<double>> Numpty::random(const size_t nodesOut, const size_t nodesIn) {
	constexpr double mean = 0;
	constexpr double standardDeviation = 1;

	std::normal_distribution distribution(mean, standardDeviation);

	vector matrix(nodesOut, vector<double>(nodesIn));

	for (size_t i = 0; i < nodesOut; i++) {
		for (size_t j = 0; j < nodesIn; j++) {
			matrix[i][j] = distribution(engine);
		}
	}

	return matrix;
}

vector<vector<double>> Numpty::xavier(const size_t nodesOut, const size_t nodesIn) {
	const double limit = std::sqrt(6.0 / (nodesIn + nodesOut));
	std::uniform_real_distribution dist(-limit, limit);

	vector matrix(nodesOut, vector<double>(nodesIn));

	for (size_t i = 0; i < nodesOut; i++) {
		for (size_t j = 0; j < nodesIn; j++) {
			matrix[i][j] = dist(engine);
		}
	}

	return matrix;
}

vector<double> Numpty::zeros(const size_t size) {
	vector<double> arr(size);

	for (size_t i = 0; i < size; i++) {
		arr[i] = 0;
	}

	return arr;
}

vector<vector<double>> Numpty::zeros(const size_t rows, const size_t columns) {
	vector matrix(rows, vector<double>(columns));

	for (size_t i = 0; i < rows; i++) {
		for (size_t j = 0; j < columns; j++) {
			matrix[i][j] = 0;
		}
	}

	return matrix;
}

vector<int> Numpty::subtractScalar(vector<int> a, const int b) {
	for (int & i : a) {
		i -= b;
	}

	return a;
}

void Numpty::addScalar(vector<int>& a, const int b) {
	for (int & i : a) {
		i += b;
	}
}

void Numpty::multiplyByScalar(vector<double>& a, const double b) {
	for (double & i : a) {
		i *= b;
	}
}

void Numpty::multiplyByScalar(vector<vector<double> > &a, const double b) {
	for (vector<double> & arr : a)
		for (double & v : arr)
			v *= b;
}

void Numpty::resetMatrixToZero(vector<vector<double>> a) {
	for (auto &row : a)
		std::ranges::fill(row, 0.0);
}

void Numpty::resetVectorToZero(std::vector<double> a) {
		std::ranges::fill(a, 0.0);
}


vector<vector<double>> Numpty::combineMatrices(const vector<vector<double>> &a, const vector<vector<double>> &b) {
	const int w = a.size();
	int h = a[0].size();

	if (w != b.size() || h != b[0].size())
		throw std::runtime_error("Matrix shapes do not correspond for likewise addition");

	vector newArr(w, vector<double>(h));
	for (int i = 0; i < w; ++i) {
		h = a[i].size();
		if (h != b[i].size())
			throw std::runtime_error("Matrix shapes do not correspond for likewise addition");
		for (int j = 0; j < h; ++j) {
			newArr[i][j] = a[i][j] + b[i][j];
		}
	}

	return newArr;
}

vector<double> Numpty::combineVectors(const vector<double> &a, const vector<double> &b) {
	const size_t n = a.size();
	if (n != b.size())
		throw std::runtime_error("Vector sizes do not correspond for likewise addition");

	vector<double> newArr(n);
	for (int i = 0; i < n; ++i) {
		newArr[i] = a[i] + b[i];
	}

	return newArr;
}

double Numpty::sum(const vector<double>& values) {
	double sum = 0;

	for (double value : values) {
		sum += value;
	}

	return sum;
}

vector<int> Numpty::argmax(const vector<vector<double>> &values) {
	vector<int> maxIndexes(values.size());

	for (size_t i = 0; i < values.size(); i++) {
		double largest = values[i][0];
		size_t largestIndex = 0;

		for (size_t j = 1; j < values[i].size(); j++) {
			if (values[i][j] > largest) {
				largest = values[i][j];
				largestIndex = j;
			}
		}

		maxIndexes[i] = largestIndex;
	}

	return maxIndexes;
}

double Numpty::dot(vector<double> a, vector<double> b) {
	double dotProduct = 0;

	for (size_t i = 0; i < a.size(); i++) {
		dotProduct += a[i] * b[i];
	}

	return dotProduct;
}

double Numpty::mean(const vector<double>& values) {
	double sum = 0;

	for (const double value : values) {
		sum += value;
	}

	return sum / values.size();
}

int Numpty::equal(const vector<int> &a, const vector<int> &b) {
	int sum = 0;

	for (size_t i = 0; i < a.size(); i++) {
		if (a[i] == b[i])
			sum++;
	}

	return sum;
}

vector<double> Numpty::logarithm(const vector<vector<double>> &inputs, const vector<int> &trueIndex) {
	vector<double> outputs(trueIndex.size());

	for (size_t i = 0; i < trueIndex.size(); i++) {
		if (inputs[i][trueIndex[i]] <= 0) throw std::runtime_error("cannot perform log on values <= 0");
		outputs[i] = std::log(inputs[i][trueIndex[i]]);
		if (std::isnan(outputs[i])) throw std::runtime_error("LOG NaN detected!");
		if (std::isinf(outputs[i])) throw std::runtime_error("LOG Inf detected!");
	}

	return outputs;
}

vector<double> Numpty::hotVectorOutput(const vector<double>& outputs, const int label, const int classesNum) {
	vector<double> values(classesNum);
	const double output = outputs[label];

	for (int i = 0; i < classesNum; ++i) {
		if (i == label)
			values[i] = output - 1;
		else
			values[i] = output;
	}

	return values;
}

vector<vector<double>> Numpty::transpose(const vector<vector<double>> a) {
	vector newMatrix(a[0].size(), vector<double>(a.size()));

	for (int i = 0; i < a.size(); ++i) {
		for (int j = 0; j < a[i].size(); ++j) {
			newMatrix[j][i] = a[i][j];
		}
	}

	return newMatrix;
}

std::vector<double> Numpty::hiddenErrors
		(const std::vector<std::vector<double> > &transposedWeights, const std::vector<double> &errorSignal) {
	vector<double> newArr(transposedWeights.size());
	for (int i = 0; i < transposedWeights.size(); ++i) {
		newArr[i] = dot(transposedWeights[i], errorSignal);
	}

	return newArr;
}


vector<vector<double>> Numpty::deepCopy2D(const vector<vector<double>> &input, const size_t l, const size_t r) {
	const size_t arrWidth = input[0].size();
	vector newArr(r - l, vector<double>(arrWidth));

	for (size_t i = 0; i < r - l; i++) {
		newArr[i] = input[i + l];
	}

	return newArr;
}

vector<int> Numpty::copy(const vector<int> &input, const size_t l, const size_t r) {
	vector<int> newArr(r - l);

	for (size_t i = 0; i < r - l; i++) {
		newArr[i] = input[i + l];
	}

	return newArr;
}