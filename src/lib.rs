//! # QuEST Rust Wrapper
//!
//! ## Introduction
//!
//! The Quantum Exact Simulation Toolkit is a high performance simulator of
//! universal quantum circuits, state-vectors and density matrices. QuEST is
//! written in C, hybridises OpenMP and MPI, and can run on a GPU. Needing only
//! compilation, QuEST is easy to run both on laptops and supercomputers (in both
//! C and C++), where it can take advantage of multicore, GPU-accelerated and
//! networked machines to quickly simulate circuits on many qubits.
//!
//! This library provides a safe wrapper around QuEST with an idiomatic Rust API.
//!
//! ## Usage
//!
//! To use quest-rs in your Rust codebase, first run:
//! ```bash
//! cargo add quest-rs
//! ```
//! or add `quest-rs` manually to your `Cargo.toml`.
//!
//! The API is simple:
//! ```
//! use quest_rs::{QuestEnv, QuReg};
//!
//! let env = QuestEnv::new();
//! let mut qubits = QuReg::new(2, &env);
//! qubits.init_plus_state().hadamard(0).controlled_not(0, 1);
//! println!(
//!     "Probability amplitude of |11> *before* measurement is: {}",
//!     qubits.probability_amplitude(0b11)
//! );
//! qubits.measure(1);
//! println!(
//!     "Probability amplitude of |11> *after* measurement is: {}",
//!     qubits.probability_amplitude(0b11)
//! );
//! ```
//!
//! The fluent API makes more complicated circuits easy to create:
//! ```
//! use quest_rs::{Complex, ComplexMatrix2, ComplexMatrixN, QReal, QuReg, QuestEnv, Vector};
//!
//! let env = QuestEnv::new();
//!
//! let mut qubits = QuReg::new(3, &env);
//! qubits.init_zero_state();
//!
//! println!("Out environment is:");
//! qubits.report_params();
//! env.report();
//!
//! // Set up the circuitry
//!
//! let unitary_alpha = Complex::new(0.5, 0.5);
//! let unitary_beta = Complex::new(0.5, -0.5);
//!
//! let unitary_matrix = ComplexMatrix2 {
//!     real: [[0.5, 0.5], [0.5, 0.5]],
//!     imag: [[0.5, -0.5], [-0.5, 0.5]],
//! };
//!
//! let mut toffoli_gate = ComplexMatrixN::new(3);
//! for i in 0..6 {
//!     toffoli_gate.set_real(i, i, 1.0);
//! }
//! toffoli_gate.set_real(6, 7, 1.0);
//! toffoli_gate.set_real(7, 6, 1.0);
//!
//! qubits
//!     .hadamard(0)
//!     .controlled_not(0, 1)
//!     .rotate_y(2, 0.1)
//!     .multi_controlled_phase_flip(vec![0, 1, 2])
//!     .unitary(0, unitary_matrix)
//!     .compact_unitary(1, unitary_alpha, unitary_beta)
//!     .rotate_around_axis(2, (3.14 / 2.0) as QReal, Vector::new(1.0, 0.0, 0.0))
//!     .controlled_compact_unitary(0, 1, unitary_alpha, unitary_beta)
//!     .multi_controlled_unitary(vec![0, 1], 2, unitary_matrix)
//!     .multi_qubit_unitary(vec![0, 1, 2], toffoli_gate);
//!
//! // Study the output
//!
//! println!("Circuit output:");
//! println!("---------------");
//! println!("Probability amplitude of |111> is: {}", qubits.probability_amplitude(0b111));
//! println!(
//!     "Probability of qubit 2 being in state 1: {}",
//!     qubits.calculate_probability_of_outcome(2, 1)
//! );
//! println!("Qubit 0 was measured in state: {}", qubits.measure(0));
//! let (outcome, outcome_probability) = qubits.measure_with_stats(2);
//! println!(
//!     "Qubit 2 collapsed to {} with probability {}",
//!     outcome, outcome_probability
//! );
//! ```
//!
//! ## Todo
//!
//! The C QuEST library has several compile-option flags which should be
//! supported using cargo features. These are:
//! - what precision to operate in (single, double or quad)
//! - whether to enable OpenMP, MPI, OpenMP+MPI or GPU
//!
//! The documentation should also be expanded to include all the relevant info
//! from the QuEST documentation.

pub mod environment;
pub mod qubits;

pub use environment::{seed_quest, seed_quest_default, QuestEnv};
pub use qubits::QuReg;

// There's currently an issue with the 128-bit integer FFI due to upstream bugs in LLVM.
// This isn't a problem when we're in single or double precision but there are hundreds
// of warnings generated because of math.h includes. This *is* a problem for quad
// precision, though. Follow these threads for more info:
// - https://github.com/rust-lang/rust/issues/54341
// - https://github.com/rust-lang/unsafe-code-guidelines/issues/119
#[allow(improper_ctypes)]

#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(dead_code)]
#[allow(clippy::all)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

// TODO: Figure out how to handle building with different precisions. Is the
// dylib built for a specific precision? If so, how can we handle this? Cargo
// features maybe?
pub type QReal = f64; // QuEST also supports f32 and f128.

// TODO: Figure out same thing but for OpenMP + MPI + GPU.

// TODO: `syncQuESTSuccess(int successCode)` also exists but it looks more like
// an internal function. Not sure whether it should be included in this wrapper.

// TODO: Is it possible to implement the `getStaticComplexMatrixN()` macro using the
// underlying function `bindArraysToStackComplexMatrixN()`?

/// Represents one complex number.
///
/// ## Examples
/// ```
/// use quest_rs::Complex;
///
/// let alpha = Complex::new(0.4, 0.6);
/// assert_eq!(alpha.real, 0.4);
/// assert_eq!(alpha.imag, 0.6);
///
/// let beta = Complex::real(1.2);
/// assert_eq!(beta.real, 1.2);
/// assert_eq!(beta.imag, 0.0);
///
/// let gamma = Complex::imag(23.9);
/// assert_eq!(gamma.real, 0.0);
/// assert_eq!(gamma.imag, 23.9);
///
/// let zero = Complex::zero();
/// assert_eq!(zero.real, 0.0);
/// assert_eq!(zero.imag, 0.0);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct Complex {
    pub real: QReal,
    pub imag: QReal,
}

impl Complex {
    /// Create a new complex number based on the real and imaginary values.
    pub fn new(real: QReal, imag: QReal) -> Self {
        Complex { real, imag }
    }

    pub fn real(real: QReal) -> Self {
        Complex { real, imag: 0.0 }
    }

    pub fn imag(imag: QReal) -> Self {
        Complex { real: 0.0, imag }
    }

    pub fn zero() -> Self {
        Complex {
            real: 0.0,
            imag: 0.0,
        }
    }
}

impl From<Complex> for ffi::Complex {
    fn from(item: Complex) -> Self {
        ffi::Complex {
            real: item.real,
            imag: item.imag,
        }
    }
}

impl From<ffi::Complex> for Complex {
    fn from(item: ffi::Complex) -> Self {
        Complex {
            real: item.real,
            imag: item.imag,
        }
    }
}

/// Represents a 2x2 matrix of complex numbers.
///
/// ## Examples
/// ```
/// use quest_rs::{Complex, ComplexMatrix2};
///
/// let pauli_x = ComplexMatrix2::new([
///     [0.0, 1.0],
///     [1.0, 0.0],
/// ], [
///     [0.0, 0.0],
///     [0.0, 0.0],
/// ]);
/// assert_eq!(pauli_x.real, [
///     [0.0, 1.0],
///     [1.0, 0.0],
/// ]);
/// assert_eq!(pauli_x.imag, [
///     [0.0, 0.0],
///     [0.0, 0.0],
/// ]);
///
/// let phase = ComplexMatrix2::compact([
///     [Complex::real(1.0), Complex::zero()],
///     [Complex::zero(), Complex::imag(1.0)],
/// ]);
/// assert_eq!(phase.real, [
///     [1.0, 0.0],
///     [0.0, 0.0],
/// ]);
/// assert_eq!(phase.imag, [
///     [0.0, 0.0],
///     [0.0, 1.0],
/// ]);
///
/// let pauli_z = ComplexMatrix2::real([
///     [1.0, 0.0],
///     [0.0, -1.0],
/// ]);
/// assert_eq!(pauli_z.real, [
///     [1.0, 0.0],
///     [0.0, -1.0],
/// ]);
/// assert_eq!(pauli_z.imag, [
///     [0.0, 0.0],
///     [0.0, 0.0],
/// ]);
///
/// let pauli_y = ComplexMatrix2::imag([
///     [0.0, -1.0],
///     [1.0, 0.0],
/// ]);
/// assert_eq!(pauli_y.real, [
///     [0.0, 0.0],
///     [0.0, 0.0],
/// ]);
/// assert_eq!(pauli_y.imag, [
///     [0.0, -1.0],
///     [1.0, 0.0],
/// ]);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct ComplexMatrix2 {
    pub real: [[f64; 2usize]; 2usize],
    pub imag: [[f64; 2usize]; 2usize],
}

impl ComplexMatrix2 {
    pub fn new(real: [[QReal; 2]; 2], imag: [[QReal; 2]; 2]) -> Self {
        ComplexMatrix2 { real, imag }
    }

    pub fn compact(values: [[Complex; 2]; 2]) -> Self {
        let mut real = [[0.0; 2]; 2];
        let mut imag = [[0.0; 2]; 2];

        for (i, row) in values.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                real[i][j] = value.real;
                imag[i][j] = value.imag;
            }
        }

        ComplexMatrix2 { real, imag }
    }

    pub fn real(real: [[QReal; 2]; 2]) -> Self {
        ComplexMatrix2 {
            real,
            imag: [[0.0, 0.0], [0.0, 0.0]],
        }
    }

    pub fn imag(imag: [[QReal; 2]; 2]) -> Self {
        ComplexMatrix2 {
            real: [[0.0, 0.0], [0.0, 0.0]],
            imag,
        }
    }
}

impl From<ComplexMatrix2> for ffi::ComplexMatrix2 {
    fn from(item: ComplexMatrix2) -> Self {
        ffi::ComplexMatrix2 {
            real: item.real,
            imag: item.imag,
        }
    }
}

/// Represents a 4x4 matrix of complex numbers.
///
/// ## Examples
/// ```
/// use quest_rs::{Complex, ComplexMatrix4};
///
/// let sqrt_swap = ComplexMatrix4::new([
///     [1.0, 0.0, 0.0, 0.0],
///     [0.0, 0.5, 0.5, 0.0],
///     [0.0, 0.5, 0.5, 0.0],
///     [0.0, 0.0, 0.0, 1.0],
/// ], [
///     [0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.5, -0.5, 0.0],
///     [0.0, -0.5, 0.5, 0.0],
///     [0.0, 0.0, 0.0, 0.0],
/// ]);
/// assert_eq!(sqrt_swap.real, [
///     [1.0, 0.0, 0.0, 0.0],
///     [0.0, 0.5, 0.5, 0.0],
///     [0.0, 0.5, 0.5, 0.0],
///     [0.0, 0.0, 0.0, 1.0],
/// ]);
/// assert_eq!(sqrt_swap.imag, [
///     [0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.5, -0.5, 0.0],
///     [0.0, -0.5, 0.5, 0.0],
///     [0.0, 0.0, 0.0, 0.0],
/// ]);
///
/// let sqrt_swap_compact = ComplexMatrix4::compact([
///     [Complex::real(1.0), Complex::zero(), Complex::zero(), Complex::zero()],
///     [Complex::zero(), Complex::new(0.5, 0.5), Complex::new(0.5, -0.5), Complex::zero()],
///     [Complex::zero(), Complex::new(0.5, -0.5), Complex::new(0.5, 0.5), Complex::zero()],
///     [Complex::zero(), Complex::zero(), Complex::zero(), Complex::real(1.0)],
/// ]);
/// assert_eq!(sqrt_swap.real, sqrt_swap_compact.real);
/// assert_eq!(sqrt_swap.imag, sqrt_swap_compact.imag);
///
/// let controlled_z = ComplexMatrix4::real([
///     [1.0, 0.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0, 0.0],
///     [0.0, 0.0, 1.0, 0.0],
///     [0.0, 0.0, 0.0, -1.0],
/// ]);
/// assert_eq!(controlled_z.real, [
///     [1.0, 0.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0, 0.0],
///     [0.0, 0.0, 1.0, 0.0],
///     [0.0, 0.0, 0.0, -1.0],
/// ]);
/// assert_eq!(controlled_z.imag, [
///     [0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0],
/// ]);
///
/// let imaginary_cnot = ComplexMatrix4::imag([
///     [1.0, 0.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 1.0],
///     [0.0, 0.0, 1.0, 0.0],
/// ]);
/// assert_eq!(imaginary_cnot.real, [
///     [0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0],
/// ]);
/// assert_eq!(imaginary_cnot.imag, [
///     [1.0, 0.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 1.0],
///     [0.0, 0.0, 1.0, 0.0],
/// ]);
/// ```
#[derive(Debug, Copy, Clone)]
pub struct ComplexMatrix4 {
    pub real: [[f64; 4usize]; 4usize],
    pub imag: [[f64; 4usize]; 4usize],
}

impl ComplexMatrix4 {
    pub fn new(real: [[QReal; 4]; 4], imag: [[QReal; 4]; 4]) -> Self {
        ComplexMatrix4 { real, imag }
    }

    pub fn compact(values: [[Complex; 4]; 4]) -> Self {
        let mut real = [[0.0; 4]; 4];
        let mut imag = [[0.0; 4]; 4];

        for (i, row) in values.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                real[i][j] = value.real;
                imag[i][j] = value.imag;
            }
        }

        ComplexMatrix4 { real, imag }
    }

    pub fn real(real: [[QReal; 4]; 4]) -> Self {
        ComplexMatrix4 {
            real,
            imag: [[0.0; 4]; 4],
        }
    }

    pub fn imag(imag: [[QReal; 4]; 4]) -> Self {
        ComplexMatrix4 {
            real: [[0.0; 4]; 4],
            imag,
        }
    }
}

impl From<ComplexMatrix4> for ffi::ComplexMatrix4 {
    fn from(item: ComplexMatrix4) -> Self {
        ffi::ComplexMatrix4 {
            real: item.real,
            imag: item.imag,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Vector {
    pub x: QReal,
    pub y: QReal,
    pub z: QReal,
}

impl Vector {
    pub fn new(x: QReal, y: QReal, z: QReal) -> Self {
        Vector { x, y, z }
    }
}

impl From<Vector> for ffi::Vector {
    fn from(item: Vector) -> Self {
        ffi::Vector {
            x: item.x,
            y: item.y,
            z: item.z,
        }
    }
}

#[derive(Debug)]
pub struct ComplexMatrixN {
    matrix: ffi::ComplexMatrixN,
    num_rows: usize,
}

// Once const generic have stabilised, we can use them here.
impl ComplexMatrixN {
    pub fn new(num_qubits: i32) -> Self {
        unsafe {
            ComplexMatrixN {
                matrix: ffi::createComplexMatrixN(num_qubits),
                num_rows: 1 << num_qubits,
            }
        }
    }

    pub fn init(&mut self, real: Vec<Vec<QReal>>, imag: Vec<Vec<QReal>>) -> &mut Self {
        let real_ptr = real
            .iter()
            .map(|row| row.as_ptr() as *mut QReal)
            .collect::<Vec<*mut QReal>>()
            .as_ptr();
        let imag_ptr = imag
            .iter()
            .map(|row| row.as_ptr() as *mut QReal)
            .collect::<Vec<*mut QReal>>()
            .as_ptr();
        unsafe {
            ffi::initComplexMatrixN(
                self.matrix,
                real_ptr as *mut *mut QReal,
                imag_ptr as *mut *mut QReal,
            );
        }
        self
    }

    pub fn display(&self) -> String {
        let mut out = String::new();
        for i in 0..self.num_rows {
            out.push_str("[");
            for j in 0..self.num_rows {
                let value = self.get(i, j);
                out.push_str(&format!("({:.5} + {:.5}j)", value.real, value.imag));
                if j != self.num_rows - 1 {
                    out.push_str("\t");
                }
            }
            out.push_str("]\n");
        }
        out
    }

    pub fn set_real(&mut self, i: usize, j: usize, value: QReal) -> &mut Self {
        self.set_value(self.matrix.real, i, j, value);
        self
    }

    pub fn set_imag(&mut self, i: usize, j: usize, value: QReal) -> &mut Self {
        self.set_value(self.matrix.imag, i, j, value);
        self
    }

    pub fn get(&self, i: usize, j: usize) -> Complex {
        Complex::new(
            self.get_value(self.matrix.real, i, j),
            self.get_value(self.matrix.imag, i, j),
        )
    }

    fn get_value(&self, raw_matrix: *mut *mut QReal, i: usize, j: usize) -> QReal {
        if i >= self.num_rows || j >= self.num_rows {
            panic!("Attempting to get value outside of bounds of complex matrix");
        }

        unsafe {
            let value_ptr = self.get_data_ptr(raw_matrix, i, j);
            *value_ptr
        }
    }

    fn set_value(&self, raw_matrix: *mut *mut QReal, i: usize, j: usize, value: QReal) {
        if i >= self.num_rows || j >= self.num_rows {
            panic!("Attempting to set value outside of bounds of complex matrix");
        }

        unsafe {
            let value_ptr = self.get_data_ptr(raw_matrix, i, j);
            *value_ptr = value;
        }
    }

    fn get_data_ptr(&self, raw_matrix: *mut *mut QReal, i: usize, j: usize) -> *mut QReal {
        unsafe {
            let row = *raw_matrix.offset(i as isize);
            let value_ptr = row.offset(j as isize);
            value_ptr
        }
    }
}

impl Drop for ComplexMatrixN {
    fn drop(&mut self) {
        unsafe {
            ffi::destroyComplexMatrixN(self.matrix);
        }
    }
}

impl From<ComplexMatrixN> for ffi::ComplexMatrixN {
    fn from(item: ComplexMatrixN) -> Self {
        item.matrix
    }
}

#[derive(Debug, Copy, Clone)]
pub enum PauliOpType {
    PauliI = 0,
    PauliX = 1,
    PauliY = 2,
    PauliZ = 3,
}

#[cfg(test)]
mod tests {
    use super::{Complex, ComplexMatrix2, ComplexMatrixN, QReal, QuReg, QuestEnv, Vector};

    #[test]
    fn two_qubit_circuit() {
        let env = QuestEnv::new();

        let mut qubits = QuReg::new(2, &env);
        qubits.init_plus_state().hadamard(0).controlled_not(0, 1);

        let prob_amp_before = qubits.probability_amplitude(0b11);
        println!(
            "Probability amplitude of |11> *before* measurement is: {}",
            prob_amp_before
        );

        qubits.measure(1);
        let prob_amp_after = qubits.probability_amplitude(0b11);
        println!(
            "Probability amplitude of |11> *after* measurement is: {}",
            prob_amp_after
        );
    }

    #[test]
    fn three_cubit_circuit() {
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

        // Also compare against values taken manually from directly running
        // equivalent C code.

        let prob_amp_state_111 = qubits.probability_amplitude(0b111);
        println!("Probability amplitude of |111> is: {}", prob_amp_state_111);
        // TODO: Assert that this probability amplitude == value from running native library code.

        let prob_qubit_two_in_state_1 = qubits.calculate_probability_of_outcome(2, 1);
        println!(
            "Probability of qubit 2 being in state 1: {}",
            prob_qubit_two_in_state_1
        );
        // TODO: Assert that this outcome probability == value from running native library code.

        println!("Qubit 0 was measured in state: {}", qubits.measure(0));
        let (outcome, outcome_probability) = qubits.measure_with_stats(2);
        println!(
            "Qubit 2 collapsed to {} with probability {}",
            outcome, outcome_probability
        );
    }
}
