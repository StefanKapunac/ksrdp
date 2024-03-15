#pragma once

#include <vector>
#include <string>
#include <random>

using neighbors_t = std::vector<std::vector<int>>;
using solution_code_t = std::vector<int>;
using solution_t = std::pair<solution_code_t, int>;

inline std::mt19937 global_random_generator(123);

neighbors_t read_instance(const std::string &file_path);

void print_neighbors(const neighbors_t &neighbors);

void print_solution(const solution_t &solution);