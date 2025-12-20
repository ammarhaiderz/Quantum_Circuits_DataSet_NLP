QUANTUM_PROBLEM_LABELS = {

    # ─────────────────────────────────────────────
    # I. Boolean & Reversible Logic
    # ─────────────────────────────────────────────
    "Boolean function synthesis":
        "Quantum circuit realizing a Boolean or reversible function",

    "Reversible circuit synthesis":
        "Synthesis of reversible logic circuits from Boolean specifications",

    "Reed–Muller circuit realization":
        "Quantum circuit implementing Reed–Muller or ESOP representations",

    "Logic network realization":
        "Quantum circuit mapped from logic networks or truth tables",

    "Permutation circuit realization":
        "Quantum circuit implementing a permutation of computational basis states",

    # ─────────────────────────────────────────────
    # II. Circuit Decomposition & Optimization
    # ─────────────────────────────────────────────
    "Gate decomposition":
        "Decomposition of multi-qubit quantum gates into elementary gates",

    "Circuit optimization":
        "Optimization of quantum circuits to reduce gate count, depth, or cost",

    "Quantum cost reduction":
        "Circuit transformations focused on reducing quantum cost",

    "Constant-depth circuit construction":
        "Construction of constant-depth or low-depth quantum circuits",

    "Gate cancellation and simplification":
        "Rules for removing redundant or canceling quantum gates",

    "Library conversion":
        "Conversion of quantum circuits between different gate libraries",

    "Ancilla-free optimization":
        "Optimization of quantum circuits without using ancilla qubits",

    "Garbage-output reduction":
        "Reduction or removal of garbage outputs in reversible circuits",

    # ─────────────────────────────────────────────
    # III. Gate Libraries & Primitive Sets
    # ─────────────────────────────────────────────
    "Quantum gate library illustration":
        "Illustration or definition of a quantum gate library",

    "MCT gate library":
        "Quantum circuits using Multiple-Controlled Toffoli gates",

    "NCV gate library":
        "Quantum circuits using NOT, CNOT, V, and V-dagger gates",

    "Clifford+T gate library":
        "Quantum circuits composed of Clifford and T gates",

    "Elementary gate construction":
        "Construction or illustration of elementary quantum gates",

    # ─────────────────────────────────────────────
    # VII. Canonical Quantum Algorithms
    # ─────────────────────────────────────────────
    "Shor's algorithm":
        "Integer factorization using quantum period finding",

    "Grover search":
        "Quantum search via amplitude amplification",

    "Deutsch–Jozsa algorithm":
        "Quantum algorithm for function classification",

    "Bernstein–Vazirani algorithm":
        "Quantum algorithm for hidden string extraction",

    "Simon's algorithm":
        "Quantum algorithm for hidden subgroup problems",

    "Quantum Fourier Transform":
        "Quantum Fourier Transform circuit",

    "Quantum phase estimation":
        "Quantum phase estimation algorithm",

    "Amplitude amplification":
        "Amplitude amplification procedure",

    # ─────────────────────────────────────────────
    # IV. State Preparation
    # ─────────────────────────────────────────────
    "Quantum state preparation":
        "Quantum circuit for preparing a specific quantum state",

    "Entangled state preparation":
        "Preparation of entangled states such as Bell or GHZ states",

    "Encoded state preparation":
        "Preparation of logical or encoded quantum states",

    # ─────────────────────────────────────────────
    # V. Measurement & Classical Control
    # ─────────────────────────────────────────────
    "Measurement and readout":
        "Quantum circuit focused on measurement and readout",

    "Classical feed-forward control":
        "Quantum circuit using measurement results for classical control",

    "Non-unitary operation implementation":
        "Quantum circuit implementing non-unitary operations using measurement",


    # ─────────────────────────────────────────────
    # VII. Error Correction & Fault Tolerance
    # ─────────────────────────────────────────────
    "Quantum error correction":
        "Quantum circuit implementing error correcting codes",

    "Syndrome measurement":
        "Quantum circuit for syndrome extraction in error correction",

    "Fault-tolerant circuit construction":
        "Fault-tolerant quantum circuit design",

    # ─────────────────────────────────────────────
    # VIII. Compilation, Mapping & Architecture
    # ─────────────────────────────────────────────
    "Quantum compilation":
        "Compilation of quantum circuits to a target gate set",

    "Qubit mapping and routing":
        "Mapping logical qubits to physical qubits with routing constraints",

    "Architecture-aware circuit design":
        "Quantum circuit designed for specific hardware connectivity",

    # ─────────────────────────────────────────────
    # IX. Simulation & Modeling
    # ─────────────────────────────────────────────
    "Quantum simulation":
        "Quantum circuit used to simulate physical or mathematical systems",

    "Hamiltonian simulation":
        "Quantum circuit simulating Hamiltonian dynamics",

    # ─────────────────────────────────────────────
    # X. Benchmarking & Illustration
    # ─────────────────────────────────────────────
    "Circuit illustration example":
        "Illustrative example of a quantum circuit",

    "Benchmark circuit":
        "Quantum circuit used for benchmarking or evaluation",

    # ─────────────────────────────────────────────
    # XI. Catch-All (IMPORTANT)
    # ─────────────────────────────────────────────
    "Unspecified quantum circuit":
        "Quantum circuit without a clearly identifiable algorithmic task",

        # ─────────────────────────────────────────────
    # XII. Variational & Hybrid Quantum Algorithms
    # ─────────────────────────────────────────────
    "Variational quantum algorithm":
        "Quantum circuit used within a variational hybrid quantum-classical loop",

    "Variational Quantum Eigensolver":
        "VQE circuit for estimating ground-state energies",

    "Quantum Approximate Optimization Algorithm":
        "QAOA circuit for solving combinatorial optimization problems",

    "Parameterized quantum circuit":
        "Parameterized circuit optimized via classical feedback",

    # ─────────────────────────────────────────────
    # XIV. Quantum Arithmetic & Oracles
    # ─────────────────────────────────────────────
    "Quantum arithmetic circuit":
        "Quantum circuit implementing arithmetic operations",

    "Quantum adder":
        "Quantum circuit for binary addition",

    "Quantum multiplier":
        "Quantum circuit for multiplication",

    "Quantum oracle construction":
        "Quantum oracle circuit encoding a classical function",



}

# to replace the abbreviations to full text for bert input

GATE_ABBREVIATIONS = {
    # Single-qubit
    "H": "Hadamard gate",
    "X": "Pauli X gate",
    "Y": "Pauli Y gate",
    "Z": "Pauli Z gate",
    "S": "Phase gate",
    "S†": "Phase dagger gate",
    "T": "T gate",
    "T†": "T dagger gate",
    "I": "Identity gate",

    # Rotations
    "RX": "Rotation X gate",
    "RY": "Rotation Y gate",
    "RZ": "Rotation Z gate",
    "Rϕ": "Phase rotation gate",
    "U": "Single qubit unitary gate",
    "U3": "Universal single qubit gate",

    # Two-qubit / multi-qubit
    "CX": "Controlled NOT gate",
    "CNOT": "Controlled NOT gate",
    "CZ": "Controlled Z gate",
    "CY": "Controlled Y gate",
    "CH": "Controlled Hadamard gate",
    "CRZ": "Controlled rotation Z gate",
    "CRX": "Controlled rotation X gate",
    "CRY": "Controlled rotation Y gate",
    "SWAP": "Swap gate",
    "CSWAP": "Controlled swap gate",
    "FREDKIN": "Controlled swap gate",

    # Multi-control
    "CCX": "Toffoli gate",
    "TOFFOLI": "Toffoli gate",
    "MCX": "Multi controlled X gate",
    "MCT": "Multi controlled Toffoli gate",

    # Measurement
    "M": "Measurement operation",
    "MEASURE": "Measurement operation",
    "RESET": "Qubit reset operation"
}


CIRCUIT_ABBREVIATIONS = {
    "QC": "Quantum circuit",
    "QEC": "Quantum error correction",
    "QFT": "Quantum Fourier transform",
    "QPE": "Quantum phase estimation",
    "VQC": "Variational quantum circuit",
    "PQC": "Parameterized quantum circuit",
    "IQP": "Instantaneous quantum polynomial-time circuit",
    "MBQC": "Measurement based quantum computation",
    "LNN": "Linear nearest neighbor architecture"
}

ALGORITHM_ABBREVIATIONS = {
    "VQE": "Variational quantum eigensolver",
    "QAOA": "Quantum approximate optimization algorithm",
    "HHL": "Harrow Hassidim Lloyd algorithm",
    "GROVER": "Grover search algorithm",
    "DJ": "Deutsch Jozsa algorithm",
    "BV": "Bernstein Vazirani algorithm",
    "SIMON": "Simon algorithm",
    "QSVT": "Quantum singular value transformation",
    "QML": "Quantum machine learning",
    "QSVM": "Quantum support vector machine",
    "QGAN": "Quantum generative adversarial network"
}

NOISE_ABBREVIATIONS = {
    "NISQ": "Noisy intermediate scale quantum",
    "T1": "Energy relaxation time",
    "T2": "Dephasing time",
    "SPAM": "State preparation and measurement",
    "QPU": "Quantum processing unit",
    "IBM-Q": "IBM quantum hardware",
    "NV": "Nitrogen vacancy center"
}


GATESET_ABBREVIATIONS = {
    "NCV": "NOT CNOT V gate library",
    "MCT": "Multi controlled Toffoli gate library",
    "CL+T": "Clifford plus T gate library",
    "CLIFFORD": "Clifford gate set"
}
