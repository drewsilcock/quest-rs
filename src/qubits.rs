use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use crate::environment::QuestEnv;
use crate::ffi;
use crate::{Complex, ComplexMatrix2, ComplexMatrix4, ComplexMatrixN, PauliOpType, QReal, Vector};

pub struct QuReg<'a> {
    env: &'a QuestEnv,
    reg: ffi::Qureg,
}

impl<'a> QuReg<'a> {
    pub fn new(num_qubits: i32, env: &'a QuestEnv) -> Self {
        unsafe {
            QuReg {
                reg: ffi::createQureg(num_qubits, env.into()),
                env,
            }
        }
    }

    pub fn new_density(num_qubits: i32, env: &'a QuestEnv) -> Self {
        unsafe {
            QuReg {
                reg: ffi::createDensityQureg(num_qubits, env.into()),
                env,
            }
        }
    }

    pub fn num_qubits(self) -> i32 {
        unsafe { ffi::getNumQubits(self.reg) }
    }

    pub fn num_prob_amplitudes(self) -> i64 {
        unsafe { ffi::getNumAmps(self.reg) }
    }

    // ---------------------
    // State Initialisations
    // ---------------------

    pub fn clone_into(&self, target_qureg: &mut QuReg) {
        unsafe {
            ffi::cloneQureg(target_qureg.reg, self.reg);
        }
    }

    pub fn init_blank_state(&mut self) -> &mut Self {
        unsafe {
            ffi::initBlankState(self.reg);
        }
        self
    }

    pub fn init_zero_state(&mut self) -> &mut Self {
        unsafe {
            ffi::initZeroState(self.reg);
        }
        self
    }

    pub fn init_plus_state(&mut self) -> &mut Self {
        unsafe {
            ffi::initPlusState(self.reg);
        }
        self
    }

    pub fn init_classical_state(&mut self, state_index: i64) -> &mut Self {
        unsafe {
            ffi::initClassicalState(self.reg, state_index);
        }
        self
    }

    pub fn init_pure_state(&mut self, pure: &QuReg) -> &mut Self {
        unsafe {
            ffi::initPureState(self.reg, pure.reg);
        }
        self
    }

    pub fn init_debug_state(&mut self) -> &mut Self {
        unsafe {
            ffi::initDebugState(self.reg);
        }
        self
    }

    pub fn init_state_from_amplitudes(
        &mut self,
        reals: Vec<QReal>,
        imags: Vec<QReal>,
    ) -> &mut Self {
        unsafe {
            ffi::initStateFromAmps(
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
            ffi::setAmps(
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
            ffi::setWeightedQureg(
                factor_one.into(),
                qureg_one.into(),
                factor_two.into(),
                qureg_two.into(),
                factor_for_this_qureg.into(),
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
            ffi::phaseShift(self.reg, target_qubit, angle);
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
            ffi::controlledPhaseShift(self.reg, qubit_one, qubit_two, angle);
        }
        self
    }

    pub fn multi_controlled_phase_shift(
        &mut self,
        control_qubits: Vec<i32>,
        angle: QReal,
    ) -> &mut Self {
        unsafe {
            ffi::multiControlledPhaseShift(
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
            ffi::controlledPhaseFlip(self.reg, qubit_one, qubit_two);
        }
        self
    }

    pub fn multi_controlled_phase_flip(&mut self, control_qubits: Vec<i32>) -> &mut Self {
        unsafe {
            ffi::multiControlledPhaseFlip(
                self.reg,
                control_qubits.as_ptr() as *mut i32,
                control_qubits.len() as i32,
            );
        }
        self
    }

    pub fn apply_s_gate(&mut self, target_qubit: i32) -> &mut Self {
        unsafe {
            ffi::sGate(self.reg, target_qubit);
        }
        self
    }

    pub fn apply_t_gate(&mut self, target_qubit: i32) -> &mut Self {
        unsafe {
            ffi::tGate(self.reg, target_qubit);
        }
        self
    }

    pub fn environment_string(&self) -> String {
        let mut env_str: Vec<c_char> = vec![0; 200];
        unsafe {
            ffi::getEnvironmentString(self.env.into(), self.reg, env_str.as_mut_ptr());
            CStr::from_ptr(env_str.as_ptr())
                .to_string_lossy()
                .into_owned()
        }
    }

    pub fn report_state_to_file(&self) -> &Self {
        unsafe {
            ffi::reportState(self.reg);
        }
        self
    }

    pub fn report_state_to_screen(&self, report_rank: i32) -> &Self {
        unsafe {
            ffi::reportStateToScreen(self.reg, self.env.into(), report_rank);
        }
        self
    }

    pub fn report_params(&self) -> &Self {
        unsafe {
            ffi::reportQuregParams(self.reg);
        }
        self
    }

    pub fn copy_state_to_gpu(&mut self) -> &mut Self {
        unsafe {
            ffi::copyStateToGPU(self.reg);
        }
        self
    }

    pub fn copy_state_from_gpu(&mut self) -> &mut Self {
        unsafe {
            ffi::copyStateFromGPU(self.reg);
        }
        self
    }

    pub fn amplitude(&self, index: i64) -> Complex {
        unsafe { ffi::getAmp(self.reg, index).into() }
    }

    pub fn real_amplitude(&self, index: i64) -> QReal {
        unsafe { ffi::getRealAmp(self.reg, index) }
    }

    pub fn imag_amplitude(&self, index: i64) -> QReal {
        unsafe { ffi::getImagAmp(self.reg, index) }
    }

    pub fn probability_amplitude(&self, index: i64) -> QReal {
        unsafe { ffi::getProbAmp(self.reg, index) }
    }

    pub fn density_amplitude(&self, row_index: i64, column_index: i64) -> Complex {
        unsafe { ffi::getDensityAmp(self.reg, row_index, column_index).into() }
    }

    pub fn calculate_total_probability(&self) -> QReal {
        unsafe { ffi::calcTotalProb(self.reg) }
    }

    pub fn calculate_purity(&self) -> QReal {
        unsafe { ffi::calcPurity(self.reg) }
    }

    pub fn calculate_fidelity(&self, pure_state: &QuReg) -> QReal {
        unsafe { ffi::calcFidelity(self.reg, pure_state.reg) }
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
            ffi::calcExpecPauliProd(
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
            ffi::calcExpecPauliSum(
                self.reg,
                pauli_operation_types.as_ptr() as *mut u32,
                term_coefficients.as_ptr() as *mut QReal,
                term_coefficients.len() as i32,
                workspace.reg,
            )
        }
    }

    pub fn calculate_probability_of_outcome(&self, measure_qubit: i32, outcome: i32) -> QReal {
        unsafe { ffi::calcProbOfOutcome(self.reg, measure_qubit, outcome) }
    }

    pub fn calculate_hilbert_schmidt_distance(&mut self, to_density_matrix: &QuReg) -> QReal {
        unsafe { ffi::calcHilbertSchmidtDistance(self.reg, to_density_matrix.reg) }
    }

    pub fn collapse_to_outcome(&mut self, measure_qubit: i32, outcome: i32) -> QReal {
        unsafe { ffi::collapseToOutcome(self.reg, measure_qubit, outcome) }
    }

    pub fn measure(&mut self, measure_qubit: i32) -> i32 {
        unsafe { ffi::measure(self.reg, measure_qubit) }
    }

    pub fn measure_with_stats(&mut self, measure_qubit: i32) -> (i32, QReal) {
        let mut outcome_probability = QReal::default();
        unsafe {
            let measurement = ffi::measureWithStats(
                self.reg,
                measure_qubit,
                &mut outcome_probability as *mut QReal,
            );

            (measurement, outcome_probability)
        }
    }

    pub fn unitary(&mut self, target_qubit: i32, unitary_matrix: ComplexMatrix2) -> &mut Self {
        unsafe {
            ffi::unitary(self.reg, target_qubit, unitary_matrix.into());
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
            ffi::compactUnitary(self.reg, target_qubit, alpha.into(), beta.into());
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
            ffi::controlledUnitary(self.reg, control_qubit, target_qubit, unitary_matrix.into());
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
            ffi::multiControlledUnitary(
                self.reg,
                control_qubits.as_ptr() as *mut i32,
                control_qubits.len() as i32,
                target_qubit,
                unitary_matrix.into(),
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
            ffi::controlledCompactUnitary(
                self.reg,
                control_qubit,
                target_qubit,
                alpha.into(),
                beta.into(),
            );
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
            ffi::twoQubitUnitary(
                self.reg,
                target_qubit_one,
                target_qubit_two,
                unitary_matrix.into(),
            );
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
            ffi::controlledTwoQubitUnitary(
                self.reg,
                control_qubit,
                target_qubit_one,
                target_qubit_two,
                unitary_matrix.into(),
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
            ffi::multiControlledTwoQubitUnitary(
                self.reg,
                control_qubits.as_ptr() as *mut i32,
                control_qubits.len() as i32,
                target_qubit_one,
                target_qubit_two,
                unitary_matrix.into(),
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
            ffi::multiQubitUnitary(
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
            ffi::controlledMultiQubitUnitary(
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
            ffi::multiControlledMultiQubitUnitary(
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
            ffi::rotateX(self.reg, qubit_to_rotate, angle);
        }
        self
    }

    pub fn rotate_y(&mut self, qubit_to_rotate: i32, angle: QReal) -> &mut Self {
        unsafe {
            ffi::rotateY(self.reg, qubit_to_rotate, angle);
        }
        self
    }

    pub fn rotate_z(&mut self, qubit_to_rotate: i32, angle: QReal) -> &mut Self {
        unsafe {
            ffi::rotateZ(self.reg, qubit_to_rotate, angle);
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
            ffi::rotateAroundAxis(self.reg, qubit_to_rotate, angle, axis.into());
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
            ffi::controlledRotateX(self.reg, control_qubit, target_qubit, angle);
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
            ffi::controlledRotateY(self.reg, control_qubit, target_qubit, angle);
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
            ffi::controlledRotateZ(self.reg, control_qubit, target_qubit, angle);
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
            ffi::controlledRotateAroundAxis(
                self.reg,
                control_qubit,
                target_qubit,
                angle,
                axis.into(),
            );
        }
        self
    }

    pub fn pauli_x(&mut self, target_qubit: i32) -> &mut Self {
        unsafe {
            ffi::pauliX(self.reg, target_qubit);
        }
        self
    }

    pub fn pauli_y(&mut self, target_qubit: i32) -> &mut Self {
        unsafe {
            ffi::pauliY(self.reg, target_qubit);
        }
        self
    }

    pub fn pauli_z(&mut self, target_qubit: i32) -> &mut Self {
        unsafe {
            ffi::pauliZ(self.reg, target_qubit);
        }
        self
    }

    pub fn hadamard(&mut self, target_qubit: i32) -> &mut Self {
        unsafe {
            ffi::hadamard(self.reg, target_qubit);
        }
        self
    }

    pub fn controlled_not(&mut self, control_qubit: i32, target_qubit: i32) -> &mut Self {
        unsafe {
            ffi::controlledNot(self.reg, control_qubit, target_qubit);
        }
        self
    }

    pub fn controlled_pauli_y(&mut self, control_qubit: i32, target_qubit: i32) -> &mut Self {
        unsafe {
            ffi::controlledPauliY(self.reg, control_qubit, target_qubit);
        }
        self
    }

    pub fn inner_product(&self, ket: QuReg) -> Complex {
        unsafe { ffi::calcInnerProduct(self.reg, ket.reg).into() }
    }

    pub fn density_inner_product(&self, other_density_matrix: QuReg) -> QReal {
        unsafe { ffi::calcDensityInnerProduct(self.reg, other_density_matrix.reg) }
    }

    pub fn start_recording_qasm(&mut self) -> &mut Self {
        unsafe {
            ffi::startRecordingQASM(self.reg);
        }
        self
    }

    pub fn stop_recording_qasm(&mut self) -> &mut Self {
        unsafe {
            ffi::stopRecordingQASM(self.reg);
        }
        self
    }

    pub fn clear_recorded_qasm(&mut self) -> &mut Self {
        unsafe {
            ffi::clearRecordedQASM(self.reg);
        }
        self
    }

    pub fn print_recorded_qasm(&mut self) -> &mut Self {
        unsafe {
            ffi::printRecordedQASM(self.reg);
        }
        self
    }

    pub fn write_recorded_qasm_to_file(&mut self, filename: &str) -> &mut Self {
        let fname_cstr = CString::new(filename).expect("Invalid filename for recorded QASM");
        unsafe {
            ffi::writeRecordedQASMToFile(self.reg, fname_cstr.as_ptr() as *mut c_char);
        }
        self
    }

    pub fn mix_dephasing(&mut self, target_qubit: i32, probability: QReal) -> &mut Self {
        unsafe {
            ffi::mixDephasing(self.reg, target_qubit, probability);
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
            ffi::mixTwoQubitDephasing(self.reg, qubit_one, qubit_two, probability);
        }
        self
    }

    pub fn mix_depolarising(&mut self, target_qubit: i32, probability: QReal) -> &mut Self {
        unsafe {
            ffi::mixDepolarising(self.reg, target_qubit, probability);
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
            ffi::mixTwoQubitDepolarising(self.reg, qubit_one, qubit_two, probability);
        }
        self
    }

    pub fn mix_damping(&mut self, target_qubit: i32, probability: QReal) -> &mut Self {
        unsafe {
            ffi::mixDamping(self.reg, target_qubit, probability);
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
            ffi::mixPauli(
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
            ffi::mixDensityMatrix(self.reg, probability, other_qureg.reg);
        }
        self
    }

    pub fn mix_kraus_map(
        &mut self,
        target_qubit: i32,
        kraus_operators: Vec<ComplexMatrix2>,
    ) -> &mut Self {
        let num_operators = kraus_operators.len();
        let ffi_kraus_operators: Vec<ffi::ComplexMatrix2> =
            kraus_operators.into_iter().map(Into::into).collect();
        unsafe {
            ffi::mixKrausMap(
                self.reg,
                target_qubit,
                ffi_kraus_operators.as_ptr() as *mut ffi::ComplexMatrix2,
                num_operators as i32,
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
        let num_operators = kraus_operators.len();
        let ffi_kraus_operators: Vec<ffi::ComplexMatrix4> =
            kraus_operators.into_iter().map(Into::into).collect();
        unsafe {
            ffi::mixTwoQubitKrausMap(
                self.reg,
                target_qubit_one,
                target_qubit_two,
                ffi_kraus_operators.as_ptr() as *mut ffi::ComplexMatrix4,
                num_operators as i32,
            );
        }
        self
    }

    pub fn mix_multi_qubit_kraus_map(
        &mut self,
        target_qubits: Vec<i32>,
        kraus_operators: Vec<ComplexMatrixN>,
    ) -> &mut Self {
        let num_operators = kraus_operators.len();
        let kraus_ops_native = kraus_operators
            .iter()
            .map(|op| op.matrix)
            .collect::<Vec<ffi::ComplexMatrixN>>();
        unsafe {
            ffi::mixMultiQubitKrausMap(
                self.reg,
                target_qubits.as_ptr() as *mut i32,
                target_qubits.len() as i32,
                kraus_ops_native.as_ptr() as *mut ffi::ComplexMatrixN,
                num_operators as i32,
            );
        }
        self
    }

    pub fn swap_gate(&mut self, qubit_one: i32, qubit_two: i32) -> &mut Self {
        unsafe {
            ffi::swapGate(self.reg, qubit_one, qubit_two);
        }
        self
    }

    pub fn sqrt_swap_gate(&mut self, qubit_one: i32, qubit_two: i32) -> &mut Self {
        unsafe {
            ffi::sqrtSwapGate(self.reg, qubit_one, qubit_two);
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
            ffi::multiStateControlledUnitary(
                self.reg,
                control_qubits.as_ptr() as *mut i32,
                control_states.as_ptr() as *mut i32,
                control_qubits.len() as i32,
                target_qubit,
                unitary_matrix.into(),
            );
        }
        self
    }

    pub fn multi_rotate_z(&mut self, target_qubits: Vec<i32>, angle: QReal) -> &mut Self {
        unsafe {
            ffi::multiRotateZ(
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
            ffi::multiRotatePauli(
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
            ffi::applyPauliSum(
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
                reg: ffi::createCloneQureg(self.reg, self.env.into()),
                env: self.env,
            }
        }
    }
}

impl Drop for QuReg<'_> {
    fn drop(&mut self) {
        unsafe { ffi::destroyQureg(self.reg, self.env.into()) }
    }
}

impl From<QuReg<'_>> for ffi::Qureg {
    fn from(item: QuReg) -> Self {
        item.reg
    }
}

impl From<&QuReg<'_>> for ffi::Qureg {
    fn from(item: &QuReg) -> Self {
        item.reg
    }
}
