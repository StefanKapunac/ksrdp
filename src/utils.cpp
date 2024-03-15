#include <fstream>
#include <sstream>
#include <iostream>

#include "utils.h"

neighbors_t read_instance(const std::string &file_path)
{
    neighbors_t neighbors;

    std::ifstream file(file_path);
    std::string line;
    for (int i = 0; std::getline(file, line); i++)
    {
        neighbors.push_back({});
        std::istringstream iss(line);
        std::string token;
        for (int j = 0; std::getline(iss, token, ','); j++)
        {
            if (std::stoi(token) == 1 && i != j)
            {
                neighbors[i].push_back(j);
            }
        }
    }

    return neighbors;
}

void print_neighbors(const neighbors_t &neighbors)
{
    for (const auto &xs : neighbors)
    {
        for (const auto &x : xs)
        {
            std::cout << x << ' ';
        }
        std::cout << std::endl;
    }
}

void print_solution(const solution_t &solution)
{
    for (const auto &s : solution.first)
    {
        std::cout << s << ' ';
    }
    std::cout << std::endl
              << solution.second << std::endl;
}