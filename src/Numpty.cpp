#include "Numpty.h"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

#include "NumptyHelper.h"


using std::vector;

vector<vector<double>> Numpty::random(const size_t nodesOut, const size_t nodesIn) {
	const unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::mt19937 engine(seed);

	constexpr double mean = 0;
	constexpr double standardDeviation = 1;

	std::normal_distribution distribution(mean, standardDeviation);

	vector arr(nodesOut, vector<double>(nodesIn));

	for (size_t i = 0; i < nodesOut; i++) {
		for (size_t j = 0; j < nodesIn; j++) {
			arr[i][j] = distribution(engine);
		}
	}

	return arr;
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
		outputs[i] = std::log(inputs[i][trueIndex[i]]);
	}

	return outputs;
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