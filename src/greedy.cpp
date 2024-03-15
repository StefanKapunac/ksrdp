#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <set>
#include <algorithm>
#include <iterator>

#include "greedy.h"

solution_t greedy_uncovered(const neighbors_t &neighbors, int num_attacks)
{
    int n = neighbors.size();
    solution_code_t solution(n, 0);
    std::set<int> covered;
    int num_covered = 0;
    for (int it = 0; num_covered < n; it++)
    {
        int next_v = -1;
        std::set<int> next_uncovered;
        int max_num_uncovered = 0;
        for (int v = 0; v < n; v++)
        {
            std::set<int> neighborhood(neighbors[v].begin(), neighbors[v].end());
            neighborhood.insert(v);
            std::set<int> uncovered;
            std::set_difference(neighborhood.begin(), neighborhood.end(),
                                covered.begin(), covered.end(),
                                std::inserter(uncovered, uncovered.begin()));

            int num_uncovered = uncovered.size();
            if (num_uncovered > max_num_uncovered || (num_uncovered == max_num_uncovered && uncovered.find(v) != uncovered.end()))
            {
                max_num_uncovered = num_uncovered;
                next_uncovered = uncovered;
                next_v = v;
            }
        }
        int next_v_not_counted;
        if (next_uncovered.find(next_v) != next_uncovered.end())
        {
            next_v_not_counted = 0;
        }
        else
        {
            next_v_not_counted = 1;
        }
        int s = std::min(num_attacks + 1, max_num_uncovered + next_v_not_counted);
        solution[next_v] = s;
        covered.insert(next_uncovered.begin(), next_uncovered.end());

        num_covered += max_num_uncovered;
    }
    int value = std::accumulate(solution.begin(), solution.end(), 0);
    return {solution, value};
}