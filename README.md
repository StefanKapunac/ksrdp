# General
This repo contains the implementation, instances and results of the paper 'Protecting graphs under multiple simultaneous attacks: a heuristic approach' by Marko Djukanovic, Stefan Kapunac, Aleksandar Kartelj and Dragan Matic.

Original implementation is in C++ and can be compiled using the ```Makefile```.
We also created Python bindings using the Pybind11 library.
For a quickstart see this [Colab notebook](https://colab.research.google.com/drive/1G_bv9X7vuZu3GsIH1XRfqo79YoIRl-Vn?usp=sharing).

# Installation
```pip install ksrdp```

# Minimal working example
```python
import ksrdp

def test_greedy(in_file_path: str, num_attacks: int) -> None:
    neighbors = ksrdp.read_instance(in_file_path)
    solution, value = ksrdp.greedy_uncovered(neighbors, num_attacks)
    print(f'greedy solution: {solution}, value: {value}')

def test_vns(in_file_path: str, num_attacks: int, vns_params: dict) -> None:
    neighbors = ksrdp.read_instance(in_file_path)
    solution, fitness, iter, end_time, best_found_time = ksrdp.vns(
        neighbors,
        vns_params['comb_take_all_bound'],
        vns_params['comb_intense_max'],
        vns_params['comb_lightweight_max'],
        num_attacks,
        vns_params['time_limit'],
        vns_params['iter_limit'],
        vns_params['k_min'],
        vns_params['k_max'],
        vns_params['move_prob'],
        vns_params['tries'],
        vns_params['num_alternatives_cutoff'],
        # verbose=True,
    )
    print(f'vns solution: {solution}, fitness: {fitness}, iter: {iter}, end_time: {end_time}, best_found_time: {best_found_time}')

def main():
    # help(ksrdp)
    in_file_path = '../instances/random/10_1.txt'
    num_attacks = 2
    test_greedy(in_file_path, num_attacks)
    vns_params = {
        'comb_take_all_bound': 100_000,
        'comb_intense_max': 10_000_000,
        'comb_lightweight_max': 10_000,
        'time_limit': 60,
        'iter_limit': 5000,
        'k_min': 1,
        'k_max': 10,
        'move_prob': 0.5,
        'tries': 10,
        'num_alternatives_cutoff': 100,
    }
    test_vns(in_file_path, num_attacks, vns_params)

if __name__ == '__main__':
    main()
```

# License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)