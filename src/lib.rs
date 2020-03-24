/// QuEST Rust Wrapper
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
pub mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub use bindings::Complex;
pub use bindings::ComplexMatrix2;
pub use bindings::ComplexMatrix4;
pub use bindings::Vector;

pub use bindings::seedQuEST as seed_quest;
pub use bindings::seedQuESTDefault as seed_quest_default;

// TODO: Figure out how to handle building with different precisions. Is the
// dylib built for a specific precision? If so, how can we handle this? Cargo
// features maybe?
type QReal = f64; // QuEST also supports f32 and f128.

// TODO: `syncQuESTSuccess(int successCode)` also exists but it looks more like
// an internal function. Not sure whether it should be included in this wrapper.

// TODO: Is it possible to implement the `getStaticComplexMatrixN()` macro using the
// underlying function `bindArraysToStackComplexMatrixN()`?

pub struct QuESTEnv {
    env: bindings::QuESTEnv,
}

impl QuESTEnv {
    pub fn new() -> Self {
        unsafe {
            QuESTEnv {
                env: bindings::createQuESTEnv(),
            }
        }
    }

    pub fn sync(&mut self) {
        unsafe {
            bindings::syncQuESTEnv(self.env);
        }
    }

    pub fn report(&self) {
        unsafe {
            bindings::reportQuESTEnv(self.env);
        }
    }
}

impl Drop for QuESTEnv {
    fn drop(&mut self) {
        unsafe {
            bindings::destroyQuESTEnv(self.env);
        }
    }
}

impl Complex {
    pub fn new(real: QReal, imag: QReal) -> Self {
        Complex { real, imag }
    }
}

impl Vector {
    pub fn new(x: QReal, y: QReal, z: QReal) -> Self {
        Vector { x, y, z }
    }
}

impl ComplexMatrix2 {
    pub fn new(real: [[QReal; 2]; 2], imag: [[QReal; 2]; 2]) -> Self {
        ComplexMatrix2 { real, imag }
    }
}

impl ComplexMatrix4 {
    pub fn new(real: [[QReal; 4]; 4], imag: [[QReal; 4]; 4]) -> Self {
        ComplexMatrix4 { real, imag }
    }
}

pub struct QuReg<'a> {
    env: &'a QuESTEnv,
    reg: bindings::Qureg,
}

impl<'a> QuReg<'a> {
    pub fn new(num_qubits: i32, env: &'a QuESTEnv) -> Self {
        unsafe {
            QuReg {
                reg: bindings::createQureg(num_qubits, env.env),
                env,
            }
        }
    }

    pub fn new_density(num_qubits: i32, env: &'a QuESTEnv) -> Self {
        unsafe {
            QuReg {
                reg: bindings::createDensityQureg(num_qubits, env.env),
                env,
            }
        }
    }

    pub fn num_qubits(self) -> i32 {
        unsafe { bindings::getNumQubits(self.reg) }
    }

    pub fn num_prob_amplitudes(self) -> i64 {
        unsafe { bindings::getNumAmps(self.reg) }
    }

    // ---------------------
    // State Initialisations
    // ---------------------

    pub fn clone_into(&self, target_qureg: &mut QuReg) {
        unsafe {
            bindings::cloneQureg(target_qureg.reg, self.reg);
        }
    }

    pub fn init_blank_state(&mut self) -> &mut Self {
        unsafe {
            bindings::initBlankState(self.reg);
        }
        self
    }

    pub fn init_zero_state(&mut self) -> &mut Self {
        unsafe {
            bindings::initZeroState(self.reg);
        }
        self
    }

    pub fn init_plus_state(&mut self) -> &mut Self {
        unsafe {
            bindings::initPlusState(self.reg);
        }
        self
    }

    pub fn init_classical_state(&mut self, state_index: i64) -> &mut Self {
        unsafe {
            bindings::initClassicalState(self.reg, state_index);
        }
        self
    }

    pub fn init_pure_state(&mut self, pure: &QuReg) -> &mut Self {
        unsafe {
            bindings::initPureState(self.reg, pure.reg);
        }
        self
    }

    pub fn init_debug_state(&mut self) -> &mut Self {
        unsafe {
            bindings::initDebugState(self.reg);
        }
        self
    }

    pub fn init_state_from_amplitudes(
        &mut self,
        reals: Vec<QReal>,
        imags: Vec<QReal>,
    ) -> &mut Self {
        unsafe {
            bindings::initStateFromAmps(
                self.reg,
                reals.as_ptr() as *mut QReal,
                imags.as_ptr() as *mut QReal,
            );
        }
        self
    }

    pub fn set_amplitudes(
        &mut self,
        start_index: i64,
        reals: Vec<QReal>,
        imags: Vec<QReal>,
    ) -> &mut Self {
        if reals.len() != imags.len() {
            panic!("Number of reals != number of amplitudes when setting qubits amplitudes.");
        }

        unsafe {
            bindings::setAmps(
                self.reg,
                start_index,
                reals.as_ptr() as *mut QReal,
                imags.as_ptr() as *mut QReal,
                reals.len() as i64,
            )
        }
        self
    }

    pub fn set_weighted_qureg(
        &mut self,
        factor_one: Complex,
        qureg_one: &QuReg,
        factor_two: Complex,
        qureg_two: &QuReg,
        factor_for_this_qureg: Complex,
    ) -> &mut Self {
        unsafe {
            bindings::setWeightedQureg(
                factor_one,
                qureg_one.reg,
                factor_two,
                qureg_two.reg,
                factor_for_this_qureg,
                self.reg,
            );
        }
        self
    }

    // ---------
    // Unitaries
    // ---------

    pub fn phase_shift(&mut self, target_qubit: i32, angle: QReal) -> &mut Self {
        unsafe {
            bindings::phaseShift(self.reg, target_qubit, angle);
        }
        self
    }

    pub fn controlled_phase_shift(
        &mut self,
        qubit_one: i32,
        qubit_two: i32,
        angle: QReal,
    ) -> &mut Self {
        unsafe {
            bindings::controlledPhaseShift(self.reg, qubit_one, qubit_two, angle);
        }
        self
    }

    pub fn multi_controlled_phase_shift(
        &mut self,
        control_qubits: Vec<i32>,
        angle: QReal,
    ) -> &mut Self {
        unsafe {
            bindings::multiControlledPhaseShift(
                self.reg,
                control_qubits.as_ptr() as *mut i32,
                control_qubits.len() as i32,
                angle,
            );
        }
        self
    }

    pub fn controlled_phase_flip(&mut self, qubit_one: i32, qubit_two: i32) -> &mut Self {
        unsafe {
            bindings::controlledPhaseFlip(self.reg, qubit_one, qubit_two);
        }
        self
    }

    pub fn multi_controlled_phase_flip(&mut self, control_qubits: Vec<i32>) -> &mut Self {
        unsafe {
            bindings::multiControlledPhaseFlip(
                self.reg,
                control_qubits.as_ptr() as *mut i32,
                control_qubits.len() as i32,
            );
        }
        self
    }

    pub fn apply_s_gate(&mut self, target_qubit: i32) -> &mut Self {
        unsafe {
            bindings::sGate(self.reg, target_qubit);
        }
        self
    }

    pub fn apply_t_gate(&mut self, target_qubit: i32) -> &mut Self {
        unsafe {
            bindings::tGate(self.reg, target_qubit);
        }
        self
    }

    pub fn environment_string(&self) -> String {
        let mut env_str: Vec<c_char> = vec![0; 200];
        unsafe {
            bindings::getEnvironmentString(self.env.env, self.reg, env_str.as_mut_ptr());
            CStr::from_ptr(env_str.as_ptr())
                .to_string_lossy()
                .into_owned()
        }
    }

    pub fn report_state_to_file(&self) -> &Self {
        unsafe {
            bindings::reportState(self.reg);
        }
        self
    }

    pub fn report_state_to_screen(&self, report_rank: i32) -> &Self {
        unsafe {
            bindings::reportStateToScreen(self.reg, self.env.env, report_rank);
        }
        self
    }

    pub fn report_params(&self) -> &Self {
        unsafe {
            bindings::reportQuregParams(self.reg);
        }
        self
    }

    pub fn copy_state_to_gpu(&mut self) -> &mut Self {
        unsafe {
            bindings::copyStateToGPU(self.reg);
        }
        self
    }

    pub fn copy_state_from_gpu(&mut self) -> &mut Self {
        unsafe {
            bindings::copyStateFromGPU(self.reg);
        }
        self
    }

    pub fn amplitude(&self, index: i64) -> Complex {
        unsafe { bindings::getAmp(self.reg, index) }
    }

    pub fn real_amplitude(&self, index: i64) -> QReal {
        unsafe { bindings::getRealAmp(self.reg, index) }
    }

    pub fn imag_amplitude(&self, index: i64) -> QReal {
        unsafe { bindings::getImagAmp(self.reg, index) }
    }

    pub fn probability_amplitude(&self, index: i64) -> QReal {
        unsafe { bindings::getProbAmp(self.reg, index) }
    }

    pub fn density_amplitude(&self, row_index: i64, column_index: i64) -> Complex {
        unsafe { bindings::getDensityAmp(self.reg, row_index, column_index) }
    }

    pub fn calculate_total_probability(&self) -> QReal {
        unsafe { bindings::calcTotalProb(self.reg) }
    }

    pub fn calculate_purity(&self) -> QReal {
        unsafe { bindings::calcPurity(self.reg) }
    }

    pub fn calculate_fidelity(&self, pure_state: &QuReg) -> QReal {
        unsafe { bindings::calcFidelity(self.reg, pure_state.reg) }
    }

    pub fn calculate_expected_pauli_product(
        &self,
        target_qubits: Vec<i32>,
        target_paulis: Vec<PauliOpType>,
        workspace: QuReg,
    ) -> QReal {
        if target_qubits.len() != target_paulis.len() {
            panic!("Number of target qubits must be the same as number of target Pauli operation types");
        }

        unsafe {
            bindings::calcExpecPauliProd(
                self.reg,
                target_qubits.as_ptr() as *mut i32,
                target_paulis.as_ptr() as *mut u32,
                target_qubits.len() as i32,
                workspace.reg,
            )
        }
    }

    pub fn calculate_expected_pauli_sum(
        &self,
        pauli_operation_types: Vec<PauliOpType>,
        term_coefficients: Vec<QReal>,
        workspace: QuReg,
    ) -> QReal {
        // There's an additional constraint that `pauli_operation_types.len() ==
        // term_coefficients.len() * qureg.num_bits_represented`, but the QuEST
        // library can handle this validation.
        unsafe {
            bindings::calcExpecPauliSum(
                self.reg,
                pauli_operation_types.as_ptr() as *mut u32,
                term_coefficients.as_ptr() as *mut QReal,
                term_coefficients.len() as i32,
                workspace.reg,
            )
        }
    }

    pub fn calculate_probability_of_outcome(&self, measure_qubit: i32, outcome: i32) -> QReal {
        unsafe { bindings::calcProbOfOutcome(self.reg, measure_qubit, outcome) }
    }

    pub fn calculate_hilbert_schmidt_distance(&mut self, to_density_matrix: &QuReg) -> QReal {
        unsafe { bindings::calcHilbertSchmidtDistance(self.reg, to_density_matrix.reg) }
    }

    pub fn collapse_to_outcome(&mut self, measure_qubit: i32, outcome: i32) -> QReal {
        unsafe { bindings::collapseToOutcome(self.reg, measure_qubit, outcome) }
    }

    pub fn measure(&mut self, measure_qubit: i32) -> i32 {
        unsafe { bindings::measure(self.reg, measure_qubit) }
    }

    pub fn measure_with_stats(&mut self, measure_qubit: i32) -> (i32, QReal) {
        let mut outcome_probability = QReal::default();
        unsafe {
            let measurement = bindings::measureWithStats(
                self.reg,
                measure_qubit,
                &mut outcome_probability as *mut QReal,
            );

            (measurement, outcome_probability)
        }
    }

    pub fn unitary(&mut self, target_qubit: i32, unitary_matrix: ComplexMatrix2) -> &mut Self {
        unsafe {
            bindings::unitary(self.reg, target_qubit, unitary_matrix);
        }
        self
    }

    pub fn compact_unitary(
        &mut self,
        target_qubit: i32,
        alpha: Complex,
        beta: Complex,
    ) -> &mut Self {
        unsafe {
            bindings::compactUnitary(self.reg, target_qubit, alpha, beta);
        }
        self
    }

    pub fn controlled_unitary(
        &mut self,
        control_qubit: i32,
        target_qubit: i32,
        unitary_matrix: ComplexMatrix2,
    ) -> &mut Self {
        unsafe {
            bindings::controlledUnitary(self.reg, control_qubit, target_qubit, unitary_matrix);
        }
        self
    }

    pub fn multi_controlled_unitary(
        &mut self,
        control_qubits: Vec<i32>,
        target_qubit: i32,
        unitary_matrix: ComplexMatrix2,
    ) -> &mut Self {
        unsafe {
            bindings::multiControlledUnitary(
                self.reg,
                control_qubits.as_ptr() as *mut i32,
                control_qubits.len() as i32,
                target_qubit,
                unitary_matrix,
            );
        }
        self
    }

    pub fn controlled_compact_unitary(
        &mut self,
        control_qubit: i32,
        target_qubit: i32,
        alpha: Complex,
        beta: Complex,
    ) -> &mut Self {
        unsafe {
            bindings::controlledCompactUnitary(self.reg, control_qubit, target_qubit, alpha, beta);
        }
        self
    }

    pub fn two_qubit_unitary(
        &mut self,
        target_qubit_one: i32,
        target_qubit_two: i32,
        unitary_matrix: ComplexMatrix4,
    ) -> &mut Self {
        unsafe {
            bindings::twoQubitUnitary(self.reg, target_qubit_one, target_qubit_two, unitary_matrix);
        }
        self
    }

    pub fn controlled_two_qubit_unitary(
        &mut self,
        control_qubit: i32,
        target_qubit_one: i32,
        target_qubit_two: i32,
        unitary_matrix: ComplexMatrix4,
    ) -> &mut Self {
        unsafe {
            bindings::controlledTwoQubitUnitary(
                self.reg,
                control_qubit,
                target_qubit_one,
                target_qubit_two,
                unitary_matrix,
            );
        }
        self
    }

    pub fn multi_controlled_two_qubit_unitary(
        &mut self,
        control_qubits: Vec<i32>,
        target_qubit_one: i32,
        target_qubit_two: i32,
        unitary_matrix: ComplexMatrix4,
    ) -> &mut Self {
        unsafe {
            bindings::multiControlledTwoQubitUnitary(
                self.reg,
                control_qubits.as_ptr() as *mut i32,
                control_qubits.len() as i32,
                target_qubit_one,
                target_qubit_two,
                unitary_matrix,
            );
        }
        self
    }

    pub fn multi_qubit_unitary(
        &mut self,
        target_qubits: Vec<i32>,
        unitary_matrix: ComplexMatrixN,
    ) -> &mut Self {
        unsafe {
            bindings::multiQubitUnitary(
                self.reg,
                target_qubits.as_ptr() as *mut i32,
                target_qubits.len() as i32,
                unitary_matrix.matrix,
            );
        }
        self
    }

    pub fn controlled_multi_qubit_unitary(
        &mut self,
        control_qubit: i32,
        target_qubits: Vec<i32>,
        unitary_matrix: ComplexMatrixN,
    ) -> &mut Self {
        unsafe {
            bindings::controlledMultiQubitUnitary(
                self.reg,
                control_qubit,
                target_qubits.as_ptr() as *mut i32,
                target_qubits.len() as i32,
                unitary_matrix.matrix,
            );
        }
        self
    }

    pub fn multi_controlled_multi_qubit_unitary(
        &mut self,
        control_qubits: Vec<i32>,
        target_qubits: Vec<i32>,
        unitary_matrix: &ComplexMatrixN,
    ) -> &mut Self {
        unsafe {
            bindings::multiControlledMultiQubitUnitary(
                self.reg,
                control_qubits.as_ptr() as *mut i32,
                control_qubits.len() as i32,
                target_qubits.as_ptr() as *mut i32,
                target_qubits.len() as i32,
                unitary_matrix.matrix,
            );
        }
        self
    }

    pub fn rotate_x(&mut self, qubit_to_rotate: i32, angle: QReal) -> &mut Self {
        unsafe {
            bindings::rotateX(self.reg, qubit_to_rotate, angle);
        }
        self
    }

    pub fn rotate_y(&mut self, qubit_to_rotate: i32, angle: QReal) -> &mut Self {
        unsafe {
            bindings::rotateY(self.reg, qubit_to_rotate, angle);
        }
        self
    }

    pub fn rotate_z(&mut self, qubit_to_rotate: i32, angle: QReal) -> &mut Self {
        unsafe {
            bindings::rotateZ(self.reg, qubit_to_rotate, angle);
        }
        self
    }

    pub fn rotate_around_axis(
        &mut self,
        qubit_to_rotate: i32,
        angle: QReal,
        axis: Vector,
    ) -> &mut Self {
        unsafe {
            bindings::rotateAroundAxis(self.reg, qubit_to_rotate, angle, axis);
        }
        self
    }

    pub fn controlled_rotate_x(
        &mut self,
        control_qubit: i32,
        target_qubit: i32,
        angle: QReal,
    ) -> &mut Self {
        unsafe {
            bindings::controlledRotateX(self.reg, control_qubit, target_qubit, angle);
        }
        self
    }

    pub fn controlled_rotate_y(
        &mut self,
        control_qubit: i32,
        target_qubit: i32,
        angle: QReal,
    ) -> &mut Self {
        unsafe {
            bindings::controlledRotateY(self.reg, control_qubit, target_qubit, angle);
        }
        self
    }

    pub fn controlled_rotate_z(
        &mut self,
        control_qubit: i32,
        target_qubit: i32,
        angle: QReal,
    ) -> &mut Self {
        unsafe {
            bindings::controlledRotateZ(self.reg, control_qubit, target_qubit, angle);
        }
        self
    }

    pub fn controlled_rotate_around_axis(
        &mut self,
        control_qubit: i32,
        target_qubit: i32,
        angle: QReal,
        axis: Vector,
    ) -> &mut Self {
        unsafe {
            bindings::controlledRotateAroundAxis(
                self.reg,
                control_qubit,
                target_qubit,
                angle,
                axis,
            );
        }
        self
    }

    pub fn pauli_x(&mut self, target_qubit: i32) -> &mut Self {
        unsafe {
            bindings::pauliX(self.reg, target_qubit);
        }
        self
    }

    pub fn pauli_y(&mut self, target_qubit: i32) -> &mut Self {
        unsafe {
            bindings::pauliY(self.reg, target_qubit);
        }
        self
    }

    pub fn pauli_z(&mut self, target_qubit: i32) -> &mut Self {
        unsafe {
            bindings::pauliZ(self.reg, target_qubit);
        }
        self
    }

    pub fn hadamard(&mut self, target_qubit: i32) -> &mut Self {
        unsafe {
            bindings::hadamard(self.reg, target_qubit);
        }
        self
    }

    pub fn controlled_not(&mut self, control_qubit: i32, target_qubit: i32) -> &mut Self {
        unsafe {
            bindings::controlledNot(self.reg, control_qubit, target_qubit);
        }
        self
    }

    pub fn controlled_pauli_y(&mut self, control_qubit: i32, target_qubit: i32) -> &mut Self {
        unsafe {
            bindings::controlledPauliY(self.reg, control_qubit, target_qubit);
        }
        self
    }

    pub fn inner_product(&self, ket: QuReg) -> Complex {
        unsafe { bindings::calcInnerProduct(self.reg, ket.reg) }
    }

    pub fn density_inner_product(&self, other_density_matrix: QuReg) -> QReal {
        unsafe { bindings::calcDensityInnerProduct(self.reg, other_density_matrix.reg) }
    }

    pub fn start_recording_qasm(&mut self) -> &mut Self {
        unsafe {
            bindings::startRecordingQASM(self.reg);
        }
        self
    }

    pub fn stop_recording_qasm(&mut self) -> &mut Self {
        unsafe {
            bindings::stopRecordingQASM(self.reg);
        }
        self
    }

    pub fn clear_recorded_qasm(&mut self) -> &mut Self {
        unsafe {
            bindings::clearRecordedQASM(self.reg);
        }
        self
    }

    pub fn print_recorded_qasm(&mut self) -> &mut Self {
        unsafe {
            bindings::printRecordedQASM(self.reg);
        }
        self
    }

    pub fn write_recorded_qasm_to_file(&mut self, filename: &str) -> &mut Self {
        let fname_cstr = CString::new(filename).expect("Invalid filename for recorded QASM");
        unsafe {
            bindings::writeRecordedQASMToFile(self.reg, fname_cstr.as_ptr() as *mut c_char);
        }
        self
    }

    pub fn mix_dephasing(&mut self, target_qubit: i32, probability: QReal) -> &mut Self {
        unsafe {
            bindings::mixDephasing(self.reg, target_qubit, probability);
        }
        self
    }

    pub fn mix_two_qubit_dephasing(
        &mut self,
        qubit_one: i32,
        qubit_two: i32,
        probability: QReal,
    ) -> &mut Self {
        unsafe {
            bindings::mixTwoQubitDephasing(self.reg, qubit_one, qubit_two, probability);
        }
        self
    }

    pub fn mix_depolarising(&mut self, target_qubit: i32, probability: QReal) -> &mut Self {
        unsafe {
            bindings::mixDepolarising(self.reg, target_qubit, probability);
        }
        self
    }

    pub fn mix_two_qubit_depolarising(
        &mut self,
        qubit_one: i32,
        qubit_two: i32,
        probability: QReal,
    ) -> &mut Self {
        unsafe {
            bindings::mixTwoQubitDepolarising(self.reg, qubit_one, qubit_two, probability);
        }
        self
    }

    pub fn mix_damping(&mut self, target_qubit: i32, probability: QReal) -> &mut Self {
        unsafe {
            bindings::mixDamping(self.reg, target_qubit, probability);
        }
        self
    }

    pub fn mix_pauli(
        &mut self,
        target_qubit: i32,
        probability_x: QReal,
        probability_y: QReal,
        probability_z: QReal,
    ) -> &mut Self {
        unsafe {
            bindings::mixPauli(
                self.reg,
                target_qubit,
                probability_x,
                probability_y,
                probability_z,
            );
        }
        self
    }

    pub fn mix_density_matrix(&mut self, probability: QReal, other_qureg: QuReg) -> &mut Self {
        unsafe {
            bindings::mixDensityMatrix(self.reg, probability, other_qureg.reg);
        }
        self
    }

    pub fn mix_kraus_map(
        &mut self,
        target_qubit: i32,
        kraus_operators: Vec<ComplexMatrix2>,
    ) -> &mut Self {
        unsafe {
            bindings::mixKrausMap(
                self.reg,
                target_qubit,
                kraus_operators.as_ptr() as *mut ComplexMatrix2,
                kraus_operators.len() as i32,
            );
        }
        self
    }

    pub fn mix_two_qubit_kraus_map(
        &mut self,
        target_qubit_one: i32,
        target_qubit_two: i32,
        kraus_operators: Vec<ComplexMatrix4>,
    ) -> &mut Self {
        unsafe {
            bindings::mixTwoQubitKrausMap(
                self.reg,
                target_qubit_one,
                target_qubit_two,
                kraus_operators.as_ptr() as *mut ComplexMatrix4,
                kraus_operators.len() as i32,
            );
        }
        self
    }

    pub fn mix_multi_qubit_kraus_map(
        &mut self,
        target_qubits: Vec<i32>,
        kraus_operators: Vec<ComplexMatrixN>,
    ) -> &mut Self {
        let kraus_ops_native = kraus_operators
            .iter()
            .map(|op| op.matrix)
            .collect::<Vec<bindings::ComplexMatrixN>>();
        unsafe {
            bindings::mixMultiQubitKrausMap(
                self.reg,
                target_qubits.as_ptr() as *mut i32,
                target_qubits.len() as i32,
                kraus_ops_native.as_ptr() as *mut bindings::ComplexMatrixN,
                kraus_operators.len() as i32,
            );
        }
        self
    }

    pub fn swap_gate(&mut self, qubit_one: i32, qubit_two: i32) -> &mut Self {
        unsafe {
            bindings::swapGate(self.reg, qubit_one, qubit_two);
        }
        self
    }

    pub fn sqrt_swap_gate(&mut self, qubit_one: i32, qubit_two: i32) -> &mut Self {
        unsafe {
            bindings::sqrtSwapGate(self.reg, qubit_one, qubit_two);
        }
        self
    }

    pub fn multi_state_controlled_unitary(
        &mut self,
        control_qubits: Vec<i32>,
        control_states: Vec<i32>,
        target_qubit: i32,
        unitary_matrix: ComplexMatrix2,
    ) -> &mut Self {
        if control_qubits.len() != control_states.len() {
            panic!(
                "The number of control qubits must be the same as the number of control states."
            );
        }

        unsafe {
            bindings::multiStateControlledUnitary(
                self.reg,
                control_qubits.as_ptr() as *mut i32,
                control_states.as_ptr() as *mut i32,
                control_qubits.len() as i32,
                target_qubit,
                unitary_matrix,
            );
        }
        self
    }

    pub fn multi_rotate_z(&mut self, target_qubits: Vec<i32>, angle: QReal) -> &mut Self {
        unsafe {
            bindings::multiRotateZ(
                self.reg,
                target_qubits.as_ptr() as *mut i32,
                target_qubits.len() as i32,
                angle,
            );
        }
        self
    }

    pub fn multi_rotate_pauli(
        &mut self,
        target_qubits: Vec<i32>,
        target_paulis: Vec<PauliOpType>,
        angle: QReal,
    ) -> &mut Self {
        if target_qubits.len() != target_paulis.len() {
            panic!("Number of target qubits must be the same as number of target Pauli operation types");
        }
        unsafe {
            bindings::multiRotatePauli(
                self.reg,
                target_qubits.as_ptr() as *mut i32,
                target_paulis.as_ptr() as *mut u32,
                target_qubits.len() as i32,
                angle,
            );
        }
        self
    }

    // ---------
    // Operators
    // ---------

    pub fn apply_pauli_sum(
        &mut self,
        pauli_operation_types: Vec<PauliOpType>,
        term_coefficients: Vec<QReal>,
    ) -> &mut Self {
        // The native function for this is something off in that you give it
        // both the input Qureg and an output Qureg and it doesn't modify the
        // input Qureg (unlike all the other functions). To normalise the for
        // this library, we do some fiddling around to make sure that this
        // method updates the current instance.
        let this_qureg_clone = self.clone();
        unsafe {
            bindings::applyPauliSum(
                this_qureg_clone.reg,
                pauli_operation_types.as_ptr() as *mut u32,
                term_coefficients.as_ptr() as *mut QReal,
                term_coefficients.len() as i32,
                self.reg,
            );
        }
        self
    }
}

impl Clone for QuReg<'_> {
    fn clone(&self) -> Self {
        unsafe {
            QuReg {
                reg: bindings::createCloneQureg(self.reg, self.env.env),
                env: self.env,
            }
        }
    }
}

impl Drop for QuReg<'_> {
    fn drop(&mut self) {
        unsafe { bindings::destroyQureg(self.reg, self.env.env) }
    }
}

pub enum PauliOpType {
    PauliI = 0,
    PauliX = 1,
    PauliY = 2,
    PauliZ = 3,
}

pub struct ComplexMatrixN {
    matrix: bindings::ComplexMatrixN,
    num_rows: usize,
}

// Once const generic have stabilised, we can use them here.
impl ComplexMatrixN {
    pub fn new(num_qubits: i32) -> Self {
        unsafe {
            ComplexMatrixN {
                matrix: bindings::createComplexMatrixN(num_qubits),
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
            bindings::initComplexMatrixN(
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
            bindings::destroyComplexMatrixN(self.matrix);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn two_qubit_circuit() {
        let env = QuESTEnv::new();

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
        let env = QuESTEnv::new();
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
            //.rotate_around_axis(2, (PI / 2) as QReal, Vector::new(1.0, 0.0, 0.0))
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
