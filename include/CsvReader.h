#pragma once

#ifndef CSVREADER_H
#define CSVREADER_H

#include <vector>
#include <string>

class CsvReader {
public:
	explicit CsvReader(const std::string& fileName);

	std::vector<std::vector<double>> getData();

	std::vector<int> getLabels();

	std::vector<std::string> getHeaders();

private:
	std::vector<std::string> headers;
	std::vector<int> labels;
	std::vector<std::vector<double>> data;
};

#endif // CSVREADER_H

