#include <iostream>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <optional>

#include "utils.h"
#include "greedy.h"
#include "vns.h"

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cout << "Usage: ./main <input file> <output file> <k>" << std::endl;
        return 1;
    }
    std::string input_file_path = argv[1];
    std::string output_file_path = argv[2];
    int attack = atoi(argv[3]);
    std::vector<int> list_attacks{attack}; //   2, 3, 4, 5};

    int tries = 10;
    unsigned num_alternatives_cutoff = 100;
    int iter_limit = 5000;
    int k_min = 1;
    int k_max = 10;
    float move_prob = 0.5;
    std::ofstream out_f(output_file_path, std::ios::app);

    neighbors_t neighbors_full = read_instance(input_file_path);
    // print_neighbors(neighbors_full);
    int final_obj_value = 0;
    bool final_feasible = true;

    // split graph into connected components and then run vns on each component separately
    std::vector<std::vector<int>> connected_components = find_connected_components(neighbors_full);
    for (auto &connected_component : connected_components)
    {
        if (connected_component.size() <= attack)
        {
            std::cout << "connected component=[";
            std::copy(connected_component.begin(), connected_component.end(), std::ostream_iterator<int>(std::cout, ", "));
            std::cout << "] smaller than num_attacks=" << attack << " ";
            std::cout << std::endl;
            out_f << std::string(50, '-') << "\ninstance=" << input_file_path << " connected component=[";
            std::copy(connected_component.begin(), connected_component.end(), std::ostream_iterator<int>(out_f, ", "));
            out_f << "] smaller than num_attacks=" << attack << " ";
            out_f << std::endl;
            final_obj_value += connected_component.size(); // all nodes must have one army
            continue;
        }

        std::sort(connected_component.begin(), connected_component.end());
        std::cout << "connected component ";
        std::copy(connected_component.begin(), connected_component.end(), std::ostream_iterator<int>(std::cout, ", "));
        std::cout << std::endl;
        neighbors_t neighbors;
        std::unordered_map<int, int> old_idx_to_new; // we want node indices to be from 0 to k
        int count = 0;
        for (int i : connected_component)
        {
            old_idx_to_new[i] = count;
            ++count;
        }
        for (int i : connected_component)
        {
            std::vector<int> row;
            for (int j : neighbors_full[i])
            {
                if (std::find(connected_component.begin(), connected_component.end(), j) != connected_component.end())
                {
                    row.push_back(old_idx_to_new[j]);
                }
            }
            neighbors.push_back(row);
        }
        // print_neighbors(neighbors);

        unsigned long long comb_take_all_bound = 1e5;
        unsigned long long comb_final_check_max = 1e9;
        unsigned long long comb_intense_max = 1e7;
        unsigned long long comb_lightweight_max = 1e4;
        int time_limit = 600;
        int scale_comb = 2;
        if (neighbors.size() < 30) // 3 groups of instances: small (< 30), medium - default (< 100), large (>= 100)
        {
            comb_take_all_bound /= scale_comb;
            comb_final_check_max /= scale_comb;
            comb_intense_max /= scale_comb;
            comb_lightweight_max /= scale_comb;
            time_limit /= 2;
        }
        else if (neighbors.size() >= 100)
        {
            comb_take_all_bound *= scale_comb;
            comb_final_check_max *= scale_comb;
            comb_intense_max *= scale_comb;
            comb_lightweight_max *= scale_comb;
            time_limit *= 2;
        }
        for (const auto &num_attacks : list_attacks)
        {
            auto [solution, obj_value, iter, time, bf_time] = vns(
                neighbors,
                comb_take_all_bound,
                comb_intense_max,
                comb_lightweight_max,
                num_attacks,
                time_limit,
                iter_limit,
                k_min,
                k_max,
                move_prob,
                tries,
                num_alternatives_cutoff);
            auto [final_infeasibility, non_defended_attack] = binary_solution_quasi_infeasibility_roulette_lazy_combinations(solution, neighbors, num_attacks, comb_final_check_max, 100ULL, num_alternatives_cutoff);
            std::ostringstream output;
            output << std::string(50, '-') << "\ninstance=" << input_file_path << " connected component=[";
            std::copy(connected_component.begin(), connected_component.end(), std::ostream_iterator<int>(output));
            output << "] num_attacks=" << num_attacks << "\nk_min=" << k_min << " k_max=" << k_max
                   << " time_limit=" << time_limit << " iter_limit=" << iter_limit << " comb_take_all_bound=" << comb_take_all_bound << " comb_final_check_max=" << comb_final_check_max
                   << " comb_intense_max=" << comb_intense_max << " comb_lightweight_max=" << comb_lightweight_max << " tries=" << tries << " move_prob=" << move_prob
                   << "\nvns obj=(" << obj_value.first << ", " << obj_value.second << ") iter=" << iter << " time=" << time << " best_found_time=" << bf_time
                   << " solution=[";
            std::copy(solution.begin(), solution.end(), std::ostream_iterator<int>(output));
            output << "] counter_example=[";
            std::copy(non_defended_attack.begin(), non_defended_attack.end(), std::ostream_iterator<int>(output));
            output << "]";
            std::string result = output.str();
            std::cout << result << std::endl;
            out_f << result << std::endl;

            final_obj_value += obj_value.second;
            if (obj_value.first > 0 || non_defended_attack.size() > 0)
            {
                final_feasible = false;
            }
        }
    }
    out_f << std::string(50, '-') << "\nfinal_feasible=" << final_feasible << " final_value=" << final_obj_value << std::endl;
    out_f.close();

    return 0;
}
