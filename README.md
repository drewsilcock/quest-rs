# QuEST Rust Wrapper

[![Build](https://github.com/drewsilcock/quest-rs/workflows/Build/badge.svg)](https://github.com/drewsilcock/quest-rs/actions?query=workflow%3ABuild)
[![Docs](https://docs.rs/quest-rs/badge.svg)](https://docs.rs/quest-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Introduction

The Quantum Exact Simulation Toolkit is a high performance simulator of
universal quantum circuits, state-vectors and density matrices. QuEST is
written in C, hybridises OpenMP and MPI, and can run on a GPU. Needing only
compilation, QuEST is easy to run both on laptops and supercomputers (in both
C and C++), where it can take advantage of multicore, GPU-accelerated and
networked machines to quickly simulate circuits on many qubits.

This library provides a safe wrapper around QuEST with an idiomatic Rust API.

## Usage

To use quest-rs in your Rust codebase, first run:
```bash
cargo add quest-rs
```
or add `quest-rs` manually to your `Cargo.toml`.

The API is simple:
```rust
use quest_rs::{QuestEnv, QuReg};

let env = QuestEnv::new();
let mut qubits = QuReg::new(2, &env);
qubits.init_plus_state().hadamard(0).controlled_not(0, 1);
println!(
    "Probability amplitude of |11> *before* measurement is: {}",
    qubits.probability_amplitude(0b11)
);
qubits.measure(1);
println!(
    "Probability amplitude of |11> *after* measurement is: {}",
    qubits.probability_amplitude(0b11)
);
```

The fluent API makes more complicated circuits easy to create:
```rust
use quest_rs::{Complex, ComplexMatrix2, ComplexMatrixN, QReal, QuReg, QuestEnv, Vector};

let env = QuestEnv::new();

let mut qubits = QuReg::new(3, &env);
qubits.init_zero_state();

println!("Out environment is:");
qubits.report_params();
env.report();

// Set up the circuitry

let unitary_alpha = Complex::new(0.5, 0.5);
let unitary_beta = Complex::new(0.5, -0.5);

let unitary_matrix = ComplexMatrix2 {
    real: [[0.5, 0.5], [0.5, 0.5]],
    imag: [[0.5, -0.5], [-0.5, 0.5]],
};

let mut toffoli_gate = ComplexMatrixN::new(3);
for i in 0..6 {
    toffoli_gate.set_real(i, i, 1.0);
}
toffoli_gate.set_real(6, 7, 1.0);
toffoli_gate.set_real(7, 6, 1.0);

qubits
    .hadamard(0)
    .controlled_not(0, 1)
    .rotate_y(2, 0.1)
    .multi_controlled_phase_flip(vec![0, 1, 2])
    .unitary(0, unitary_matrix)
    .compact_unitary(1, unitary_alpha, unitary_beta)
    .rotate_around_axis(2, (3.14 / 2.0) as QReal, Vector::new(1.0, 0.0, 0.0))
    .controlled_compact_unitary(0, 1, unitary_alpha, unitary_beta)
    .multi_controlled_unitary(vec![0, 1], 2, unitary_matrix)
    .multi_qubit_unitary(vec![0, 1, 2], toffoli_gate);

// Study the output

println!("Circuit output:");
println!("---------------");
println!("Probability amplitude of |111> is: {}", qubits.probability_amplitude(0b111));
println!(
    "Probability of qubit 2 being in state 1: {}",
    qubits.calculate_probability_of_outcome(2, 1)
);
println!("Qubit 0 was measured in state: {}", qubits.measure(0));
let (outcome, outcome_probability) = qubits.measure_with_stats(2);
println!(
    "Qubit 2 collapsed to {} with probability {}",
    outcome, outcome_probability
);
```

## Todo

The C QuEST library has several compile-option flags which should be
supported using cargo features. These are:
- what precision to operate in (single, double or quad)
- whether to enable OpenMP, MPI, OpenMP+MPI or GPU

The documentation should also be expanded to include all the relevant info
from the QuEST documentation.
