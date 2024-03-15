#include <algorithm>
#include <random>
#include <cassert>
#include <chrono>
#include <iostream>
#include <queue>
#include <optional>
#include <iterator>
#include <numeric>
#include <unordered_set>
#include <set>

#include "vns.h"

solution_t initialize(const neighbors_t &neighbors, int num_attacks)
{
    return greedy_uncovered(neighbors, num_attacks);
    // solution_code_t solution(neighbors.size(), 1); // trivial solution
    // return {solution, neighbors.size()};
}

solution_code_t shaking(const solution_code_t &solution, int k)
{
    solution_code_t new_solution = solution;
    std::vector<int> solution_multiset;
    int n = solution.size();
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < solution[i]; ++j)
        {
            solution_multiset.push_back(i);
        }
    }
    std::vector<int> chosen_nodes;
    std::sample(solution_multiset.begin(), solution_multiset.end(),
                std::back_inserter(chosen_nodes),
                k,
                global_random_generator);
    for (const auto &node : chosen_nodes)
    {
        new_solution[node] -= 1;
    }
    std::uniform_int_distribution<int> dist(0, n - 1);
    int v;
    for (int i = 0; i < k; ++i)
    {
        v = dist(global_random_generator);
        new_solution[v] += 1;
    }
    // reduce one random army
    do
    {
        v = dist(global_random_generator);
    } while (new_solution[v] == 0);
    new_solution[v] -= 1;

    int value = std::accumulate(solution.begin(), solution.end(), 0);
    int new_value = std::accumulate(new_solution.begin(), new_solution.end(), 0);
    assert(new_value + 1 == value);

    return new_solution;
}

std::pair<bool, node_ids_t> is_attack_defended(const solution_code_t &solution, const neighbors_t &neighbors, const std::vector<int> &attack, int tries, unsigned num_alternatives_cutoff)
{
    int attack_size = attack.size();
    bool attack_defended = false;
    node_ids_t defending_nodes_default;
    defending_nodes_default.reserve(attack_size);
    std::unordered_map<int, int> node_that_gave_help_default;
    node_that_gave_help_default.reserve(attack_size);
    node_ids_t reduced_attack;
    reduced_attack.reserve(attack_size);
    std::vector<int> candidates;
    candidates.reserve(std::accumulate(solution.begin(), solution.end(), 0));
    std::vector<std::vector<int>> all_alternatives;
    all_alternatives.push_back({});
    int product = 1;
    for (const auto &v : attack)
    {
        if (solution[v] > 0)
        {
            defending_nodes_default.insert(v);
            continue;
        }
        node_ids_t alternatives;
        for (const auto &u : neighbors[v])
        {
            int diff;
            if (node_that_gave_help_default.find(u) != node_that_gave_help_default.end())
            {
                diff = solution[u] - node_that_gave_help_default[u] - 1;
            }
            else
            {
                diff = solution[u] - 1;
            }
            if (diff > 0)
            {
                alternatives.insert(u);
            }
        }
        if (alternatives.size() == 0)
        {
            return {false, {}};
        }
        else
        {
            product *= alternatives.size();
            reduced_attack.insert(v);
            std::vector<std::vector<int>> new_all_alternatives;
            for (const auto &alt : all_alternatives)
            {
                for (auto elem : alternatives)
                {
                    auto alt_copy = alt;
                    alt_copy.push_back(elem);
                    new_all_alternatives.push_back(alt_copy);
                }
            }
            all_alternatives = new_all_alternatives;
        }
    }

    if (all_alternatives.size() < num_alternatives_cutoff)
    {
        for (unsigned i = 0; i < all_alternatives.size(); ++i)
        {
            attack_defended = true;

            auto node_that_gave_help = node_that_gave_help_default;
            auto defending_nodes = defending_nodes_default;

            int k = 0;
            for (const auto &v : reduced_attack)
            {
                assert(solution[v] == 0);
                int u = all_alternatives[i][k];
                assert(solution[u] > 1);
                int helped_count = 0;
                if (node_that_gave_help.find(u) != node_that_gave_help.end())
                {
                    helped_count = node_that_gave_help.at(u);
                }
                int diff = solution[u] - helped_count - 1;
                if (diff == 0)
                {
                    attack_defended = false;
                    break;
                }
                ++node_that_gave_help[u];
                defending_nodes.insert(u);
                ++k;
            }
            if (attack_defended)
            {
                return {attack_defended, defending_nodes};
            }
        }
        return {false, {}};
    }
    else
    {
        while (tries > 0)
        {
            attack_defended = true;
            --tries;

            auto node_that_gave_help = node_that_gave_help_default;
            auto defending_nodes = defending_nodes_default;

            for (const auto &v : reduced_attack)
            {
                assert(solution[v] == 0);
                candidates.clear();
                int neighbors_v_size = neighbors[v].size();
                for (int j = 0; j < neighbors_v_size; ++j)
                {
                    int u = neighbors[v][j];
                    if (solution[u] <= 1)
                    {
                        continue;
                    }
                    int helped_count = 0;
                    if (node_that_gave_help.find(u) != node_that_gave_help.end())
                    {
                        helped_count = node_that_gave_help.at(u);
                    }
                    int diff = solution[u] - helped_count - 1;
                    for (int i = 0; i < diff; ++i)
                    {
                        candidates.push_back(u);
                    }
                }
                int num_candidates = candidates.size();
                if (num_candidates == 0)
                {
                    attack_defended = false;
                    break;
                }
                else
                {
                    std::uniform_int_distribution<int> dist(0, num_candidates - 1);
                    int selected_u = candidates[dist(global_random_generator)];
                    ++node_that_gave_help[selected_u];
                    defending_nodes.insert(selected_u);
                }
            }
            if (attack_defended)
            {
                return {attack_defended, defending_nodes};
            }
        }
        return {false, {}};
    }
}

std::pair<int, std::vector<int>> binary_solution_quasi_infeasibility_roulette_lazy_combinations(const solution_code_t &solution, const neighbors_t &neighbors, int num_attacks, unsigned long long max_attack_comb, int tries, unsigned num_alternatives_cutoff, int log_per_attacks)
{
    int h = 0;
    int num_nodes = neighbors.size();
    auto comb_count = calc_num_combinations(num_nodes, num_attacks);
    std::cout << "Checking " << std::min(comb_count, max_attack_comb) << " out of " << max_attack_comb << std::endl;
    int mod = comb_count / max_attack_comb;
    if (mod == 0)
    {
        mod = 1;
    }
    int checked = 0;
    std::vector<int> nodes(num_nodes);
    std::iota(nodes.begin(), nodes.end(), 0);
    auto combs = generate_combinations_new(nodes, num_attacks);
    for (const auto &attack : combs)
    {
        if (h % mod != 0)
        {
            ++h;
            continue;
        }
        auto [attack_defended, dummy_defending_nodes] = is_attack_defended(solution, neighbors, attack, tries, num_alternatives_cutoff);
        ++checked;
        if (checked % log_per_attacks == 0)
        {
            std::cout << "Checked " << checked << std::endl;
        }
        if (!attack_defended)
        {
            return {1, attack};
        }
        ++h;
    }
    std::vector<int> v;
    return {0, v};
}

std::tuple<int, coverage_info_t, node_ids_t> solution_quasi_infeasibility_roulette(const solution_code_t &solution, const neighbors_t &neighbors, const std::vector<std::vector<int>> &predefined_attacks, int tries, unsigned num_alternatives_cutoff, bool create_coverage_info)
{
    int non_defended_attacks_count = 0;
    node_ids_t non_defended_attack_ids;
    coverage_info_t coverage_info;
    if (create_coverage_info)
    {
        int num_nodes = neighbors.size();
        for (int i = 0; i < num_nodes; ++i)
        {
            coverage_info.emplace(i, node_ids_t());
        }
    }
    int num_attacks = predefined_attacks.size();
    for (int h = 0; h < num_attacks; ++h)
    {
        auto [attack_defended, defending_nodes] = is_attack_defended(solution, neighbors, predefined_attacks[h], tries, num_alternatives_cutoff);
        if (!attack_defended)
        {
            ++non_defended_attacks_count;
            non_defended_attack_ids.insert(h);
        }
        else
        {
            if (create_coverage_info)
            {
                for (const auto &v : defending_nodes)
                {
                    coverage_info[v].insert(h);
                }
            }
        }
    }

    return {non_defended_attacks_count, coverage_info, non_defended_attack_ids};
}

std::pair<solution_code_t, fitness_t> local_search_swap_first_impr(const solution_code_t &solution, const neighbors_t &neighbors, const std::vector<std::vector<int>> &attacks, int tries, unsigned num_alternatives_cutoff)
{
    solution_code_t new_solution = solution;
    auto quasi_feasibility = solution_quasi_infeasibility_roulette(solution, neighbors, attacks, tries, num_alternatives_cutoff);
    int infeas = std::get<0>(quasi_feasibility);
    int n = neighbors.size();
    int num_attacks = attacks.size();
    bool improved = true;
    while (improved && infeas > 0)
    {
        auto [dummy_infeas, coverage_info, non_defended_attacks_ids] = solution_quasi_infeasibility_roulette(new_solution, neighbors, attacks, tries, num_alternatives_cutoff, true);
        improved = false;
        std::vector<std::pair<int, int>> different_node_pairs;
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < i; ++j)
            {
                different_node_pairs.emplace_back(i, j);
            }
        }
        std::shuffle(different_node_pairs.begin(), different_node_pairs.end(), global_random_generator);
        for (const auto &[node1, node2] : different_node_pairs)
        {
            int old1 = new_solution[node1];
            int old2 = new_solution[node2];
            node_ids_t node1_coverage = coverage_info[node1];
            node_ids_t node2_coverage = coverage_info[node2];
            int tot = old1 + old2;
            for (int new1 = 0; new1 <= tot; ++new1)
            {
                if (new1 == old1)
                {
                    continue;
                }
                new_solution[node1] = new1;
                new_solution[node2] = tot - new1;
                std::vector<std::vector<int>> reduced_attacks;
                for (int h = 0; h < num_attacks; ++h)
                {
                    if (non_defended_attacks_ids.find(h) != non_defended_attacks_ids.end() || (new_solution[node1] < old1 && node1_coverage.find(h) != node1_coverage.end()) || (new_solution[node2] < old2 && node2_coverage.find(h) != node2_coverage.end()))
                    {
                        reduced_attacks.push_back(attacks[h]);
                    }
                }
                auto [new_infeas, dummy_coverage_info, dummy_non_defended_attacks_ids] = solution_quasi_infeasibility_roulette(new_solution, neighbors, reduced_attacks, tries, num_alternatives_cutoff);
                if (new_infeas < infeas)
                {
                    infeas = new_infeas;
                    improved = true;
                    break;
                }
            }
            if (improved)
            {
                break;
            }
            else
            {
                new_solution[node1] = old1;
                new_solution[node2] = old2;
            }
        }
    }
    fitness_t new_fitness{infeas, std::accumulate(new_solution.begin(), new_solution.end(), 0)};
    return {new_solution, new_fitness};
}

unsigned long long calc_num_combinations(int n, int k)
{
    unsigned long long result = 1;
    for (int i = 0; i < k; ++i)
    {
        result *= (n - i);
        result /= (i + 1);
    }
    return result;
}

template <typename T>
std::vector<std::vector<typename T::value_type>> generate_combinations_new(const T &xs, int k)
{
    int n = xs.size();
    if (n < k)
    {
        return {};
    }
    std::vector<int> indices(k);
    std::iota(indices.begin(), indices.end(), 0);
    auto num_possible_combs = calc_num_combinations(n, k);
    std::vector<std::vector<typename T::value_type>> result;
    result.reserve(num_possible_combs);
    std::vector<typename T::value_type> combination;
    combination.reserve(k);
    for (int idx : indices)
    {
        combination.push_back(xs[idx]);
    }
    result.push_back(combination);
    for (int iter = 1; true; ++iter)
    {
        int i = k - 1;
        while (i >= 0 && indices[i] == n - k + i)
        {
            --i;
        }
        if (i < 0)
        {
            break;
        }
        ++indices[i];
        for (int j = i + 1; j < k; ++j)
        {
            indices[j] = indices[j - 1] + 1;
        }
        for (int j = i; j < k; ++j) // elements before ith index are the same as in the previous combination
        {
            combination[j] = xs[indices[j]];
        }
        result.push_back(combination);
    }
    return result;
}

std::vector<int> bfs(int v, std::vector<bool> &visited, const neighbors_t &neighbors)
{
    std::vector<int> component;
    std::queue<int> q;
    q.push(v);
    visited[v] = true;
    while (!q.empty())
    {
        int node = q.front();
        q.pop();
        component.push_back(node);
        for (int u : neighbors[node])
        {
            if (!visited[u])
            {
                visited[u] = true;
                q.push(u);
            }
        }
    }
    return component;
}

std::vector<std::vector<int>> find_connected_components(const neighbors_t &neighbors)
{
    int n = neighbors.size();
    std::vector<bool> visited(n, false);
    std::vector<std::vector<int>> connected_components;
    for (int v = 0; v < n; ++v)
    {
        if (!visited[v])
        {
            connected_components.push_back(bfs(v, visited, neighbors));
        }
    }
    return connected_components;
}

class hasher
{
public:
    std::size_t operator()(std::vector<int> const &vec) const
    {
        std::size_t seed = vec.size();
        for (auto x : vec)
        {
            x = ((x >> 16) ^ x) * 0x45d9f3b;
            x = ((x >> 16) ^ x) * 0x45d9f3b;
            x = (x >> 16) ^ x;
            seed ^= x + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

std::vector<std::vector<int>> make_attacks_Ns(int n, int k, const neighbors_t &neighbors, int s)
{
    if (s > 3 || s < 1)
    {
        throw std::out_of_range("Allowed neighborhood sizes (s) are 1, 2 or 3");
    }
    auto num_possible_combs = calc_num_combinations(n, k);
    std::vector<int> critical_nodes;
    auto connected_components = find_connected_components(neighbors);
    for (const auto &c : connected_components)
    {
        if ((int)c.size() < k)
        {
            for (int v : c)
            {
                critical_nodes.push_back(v);
            }
        }
    }
    std::unordered_set<std::vector<int>, hasher> attacks;
    for (int v = 0; v < n; ++v)
    {
        std::set<int> ns_v; // here we need sorted set, so that combinations are always sorted the same way
        ns_v.insert(v);
        for (const auto &u : neighbors[v])
        {
            ns_v.insert(u);
            if (s > 1)
            {
                for (const auto &w : neighbors[u])
                {
                    ns_v.insert(w);
                    if (s > 2)
                    {
                        ns_v.insert(neighbors[w].begin(), neighbors[w].end());
                    }
                }
            }
        }
        // add isolated nodes to each neighborhood
        ns_v.insert(critical_nodes.begin(), critical_nodes.end());
        std::vector<int> vec(ns_v.begin(), ns_v.end());
        auto combs = generate_combinations_new(vec, k);
        attacks.insert(combs.begin(), combs.end());

        if (attacks.size() == num_possible_combs)
        {
            break;
        }
    }
    std::cout << "There are " << attacks.size() << " attacks for neighborhood " << s << " out of possible " << num_possible_combs << std::endl;
    std::vector<std::vector<int>> result(attacks.begin(), attacks.end());
    return result;
}

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
    bool verbose)
{
    auto start_time = std::chrono::steady_clock::now();
    int num_nodes = neighbors.size();
    auto comb_count = calc_num_combinations(num_nodes, num_attacks);
    std::vector<std::vector<int>> attacks_intense;
    std::vector<std::vector<int>> attacks_lightweight;
    if (comb_count <= comb_take_all_bound)
    {
        std::vector<int> nodes(num_nodes);
        std::iota(nodes.begin(), nodes.end(), 0);
        attacks_intense = generate_combinations_new(nodes, num_attacks);
        attacks_lightweight = attacks_intense;
    }
    else
    {
        attacks_intense = make_attacks_Ns(num_nodes, num_attacks, neighbors, 3);
        if (attacks_intense.size() <= comb_take_all_bound)
        {
            attacks_lightweight = attacks_intense;
        }
        else
        {
            attacks_lightweight = make_attacks_Ns(num_nodes, num_attacks, neighbors, 1);
        }
    }
    if (attacks_intense.size() > comb_intense_max)
    {
        std::sample(attacks_intense.begin(), attacks_intense.end(),
                    attacks_intense.begin(),
                    comb_intense_max,
                    global_random_generator);
        attacks_intense.resize(comb_intense_max);
    }
    if (attacks_lightweight.size() > comb_lightweight_max)
    {
        std::sample(attacks_lightweight.begin(), attacks_lightweight.end(),
                    attacks_lightweight.begin(),
                    comb_lightweight_max,
                    global_random_generator);
        attacks_lightweight.resize(comb_lightweight_max);
    }

    auto [solution, value] = initialize(neighbors, num_attacks);
    auto [infeasibility, dummy_coverage_info, dummy_node_ids] = solution_quasi_infeasibility_roulette(solution, neighbors, attacks_intense, tries, num_alternatives_cutoff);
    fitness_t fitness{infeasibility, value};
    if (verbose)
    {
        std::cout << "Initial solution fitness: " << fitness.first << ' ' << fitness.second << std::endl;
    }
    int iter = 0;
    float best_found_time = 0;
    while (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() / 1000.0 < time_limit && iter < iter_limit)
    {
        for (int k = k_min; k <= k_max; ++k)
        {
            ++iter;
            if (k > fitness.second || std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() / 1000.0 > time_limit || iter > iter_limit)
            {
                break;
            }
            auto shaked_solution = shaking(solution, k);
            auto [new_solution, new_fitness] = local_search_swap_first_impr(shaked_solution, neighbors, attacks_lightweight, tries, num_alternatives_cutoff);
            if (verbose)
            {
                std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() / 1000.0
                          << "s iter: " << iter << " k: " << k << " fit: " << fitness.first << ' ' << fitness.second
                          << " new_fit: " << new_fitness.first << ' ' << new_fitness.second << std::endl;
            }
            std::uniform_real_distribution<float> real_dist(0.0, 1.0);
            bool fit_cmp = new_fitness < fitness;
            if (fit_cmp || (new_fitness == fitness && real_dist(global_random_generator) < move_prob))
            {
                auto [test_infeas, dummy1, dummy2] = solution_quasi_infeasibility_roulette(new_solution, neighbors, attacks_intense, tries, num_alternatives_cutoff);
                if (verbose)
                {
                    std::cout << "after test: " << test_infeas << std::endl;
                }
                if (test_infeas == 0)
                {
                    fitness = new_fitness;
                    solution = new_solution;
                    if (fit_cmp)
                    {
                        best_found_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() / 1000.0;
                    }
                    break;
                }
            }
        }
    }
    auto end_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_time).count() / 1000.0;
    return {solution, fitness, iter, end_time, best_found_time};
}
