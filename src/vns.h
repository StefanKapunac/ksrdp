#pragma once

#include <random>
#include <unordered_set>
#include <unordered_map>

#include "utils.h"
#include "greedy.h"

using fitness_t = std::pair<int, int>;
using node_ids_t = std::unordered_set<int>;
using coverage_info_t = std::unordered_map<int, node_ids_t>;
using vns_result_t = std::tuple<solution_code_t, fitness_t, int, float, float>;

solution_t initialize(const neighbors_t &neighbors, int num_attacks);

solution_code_t shaking(const solution_code_t &solution, int k);

std::tuple<int, coverage_info_t, node_ids_t> solution_quasi_infeasibility_roulette(const solution_code_t &solution, const neighbors_t &neighbors, const std::vector<std::vector<int>> &attacks, int tries, unsigned num_alternatives_cutoff, bool create_coverage_info = false);

std::pair<int, std::vector<int>> binary_solution_quasi_infeasibility_roulette_lazy_combinations(const solution_code_t &solution, const neighbors_t &neighbors, int num_attacks, unsigned long long max_attack_comb, int tries, unsigned num_alternatives_cutoff, int log_per_attacks = 1e6);

std::pair<solution_code_t, fitness_t> local_search_swap_first_impr(const solution_code_t &solution, const neighbors_t &neighbors, const std::vector<std::vector<int>> &attacks, int tries, unsigned num_alternatives_cutoff);

std::pair<bool, node_ids_t> is_attack_defended(const solution_code_t &solution, const neighbors_t &neighbors, const std::vector<int> &attack, int tries, unsigned num_alternatives_cutoff);

std::vector<std::vector<int>> make_attacks_Ns(int n, int k, const neighbors_t &neighbors, int s = 3);

unsigned long long calc_num_combinations(int n, int k);

std::vector<std::vector<int>> find_connected_components(const neighbors_t &neighbors);

template <typename T>
std::vector<std::vector<typename T::value_type>> generate_combinations_new(const T &xs, int k);

vns_result_t vns(
    const neighbors_t &neighbors,
    unsigned long long comb_take_all_bound,
    unsigned long long comb_intense_max,
    unsigned long long comb_lightweight_max,
    int num_attacks,
    int time_limit,
    int iter_limit,
    int k_min,
    int k_max,
    float move_prob,
    int tries,
    unsigned num_alternatives_cutoff,
    bool verbose = false);