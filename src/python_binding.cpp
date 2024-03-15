#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "vns.h"

namespace py = pybind11;

PYBIND11_MODULE(ksrdp, m)
{
    m.def("read_instance", &read_instance, "Read instance from the given file_path",
          py::arg("file_path"));
    m.def("greedy_uncovered", &greedy_uncovered, "Greedy algorithm based on node coverage",
          py::arg("neighbors"), py::arg("num_attacks"));
    m.def("initialize", &initialize, "Initialize the solution of VNS using the greedy algorithm",
          py::arg("neighbors"), py::arg("num_attacks"));
    m.def("shaking", &shaking, "Shaking operator",
          py::arg("solution"), py::arg("k"));
    m.def("solution_quasi_infeasibility_roulette", &solution_quasi_infeasibility_roulette, "Probabilistic quasi-infeasibility verification",
          py::arg("solution"), py::arg("neighbors"), py::arg("predefined_attacks"), py::arg("tries"), py::arg("num_alternatives_cutoff"), py::arg("create_coverage_info") = false);
    m.def("binary_solution_quasi_infeasibility_roulette_lazy_combinations", &binary_solution_quasi_infeasibility_roulette_lazy_combinations, "Binary quasi-infeasibility verification",
          py::arg("solution"), py::arg("neighbors"), py::arg("num_attacks"), py::arg("max_attack_comb"), py::arg("tries"), py::arg("num_alternatives_cutoff"), py::arg("log_per_attacks") = 1000000);
    m.def("local_search_swap_first_impr", &local_search_swap_first_impr, "Local search based on swaps with first improvement strategy",
          py::arg("solution"), py::arg("neighbors"), py::arg("attacks"), py::arg("tries"), py::arg("num_alternatives_cutoff"));
    m.def("is_attack_defended", &is_attack_defended, "Check if the given attack is defended with the given solution",
          py::arg("solution"), py::arg("neighbors"), py::arg("attack"), py::arg("tries"), py::arg("num_alternatives_cutoff"));
    m.def("make_attacks_Ns", &make_attacks_Ns, "Generate attacks based on s-th neighborhood in the graph",
          py::arg("n"), py::arg("k"), py::arg("neighbors"), py::arg("s") = 3);
    m.def("calc_num_combinations", &calc_num_combinations, "Calculate the number of possible combinations n over k",
          py::arg("n"), py::arg("k"));
    m.def("find_connected_components", &find_connected_components, "Find connected components in the graph",
          py::arg("neighbors"));
    m.def("generate_combinations_new", &generate_combinations_new<std::vector<int>>, "Generate combinations of length k of elements from xs",
          py::arg("xs"), py::arg("k"));
    m.def("vns", &vns, "VNS algorithm",
          py::arg("neighbors"),
          py::arg("comb_take_all_bound"),
          py::arg("comb_intense_max"),
          py::arg("comb_lightweight_max"),
          py::arg("num_attacks"),
          py::arg("time_limit"),
          py::arg("iter_limit"),
          py::arg("k_min"),
          py::arg("k_max"),
          py::arg("move_prob"),
          py::arg("tries"),
          py::arg("num_alternatives_cutoff"),
          py::arg("verbose") = false);
}