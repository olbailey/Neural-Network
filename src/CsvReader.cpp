#include "CsvReader.h"

#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>

using std::vector;


CsvReader::CsvReader(const std::string& fileName) {
	std::ifstream myFile(fileName);

	if (!myFile.is_open()) throw std::runtime_error("File not found");
	if (!myFile.good()) throw std::runtime_error("File not good?");

	//helper variables
	std::string line, columnName;
	std::string value;

	std::getline(myFile, line);

	std::stringstream headerStream(line);

	while (std::getline(headerStream, columnName, ',')) {
		headers.push_back(columnName);
	}

	while (std::getline(myFile, line)) {
		std::stringstream valueStream(line);

		std::getline(valueStream, value, ',');
		labels.push_back(std::stoi(value));

		std::vector<double> rowValues;

		while (std::getline(valueStream, value, ',')) {
			rowValues.push_back(std::stod(value));
		}

		data.push_back(rowValues);
	}
}

vector<vector<double>> CsvReader::getData() { return data; }

vector<int> CsvReader::getLabels() { return labels; }

vector<std::string> CsvReader::getHeaders() { return headers; }