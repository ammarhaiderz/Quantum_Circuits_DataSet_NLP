"""
Query definitions and token configurations.
"""

from nltk.stem import PorterStemmer

STEMMER = PorterStemmer()

# ================ QUERY SETS ================
# QUERY_SETS = {
#     "circuit_core": """
#         quantum circuit diagram
#         qubit circuit
#         gate based quantum circuit
#         gate sequence
#         controlled gate
#         qubit register
#         circuit depth
#     """,

#     "gate_level": """
#         cnot hadamard pauli
#         rx ry rz
#         controlled not
#         control target qubit
#         multi qubit gate
#         ancilla qubit
#     """,

#     "algorithmic_circuits": """
#         quantum algorithm 
#         oracle
#         grover 
#         shor 
#         qft
#     """,

#     "variational_circuits": """
#         variational quantum circuit
#         parameterized circuit
#         ansatz circuit
#         vqe circuit
#         qaoa circuit
#     """
# }

QUERY_SETS = {
    "circuit_fundamentals": """
        quantum circuit diagram
        circuit with gates
        qubit circuit
        gate based quantum circuit
        gate sequence
        controlled gate
        qubit register
        circuit depth
        multi-qubit gate
        single-qubit rotation
        entangling gate
        two-qubit gate
        circuit schematic
        quantum gate circuit
        qubit connectivity
        circuit topology
    """,

    "quantum_algorithms": """
        ghz
        bell state
        Josephson tunnel junction
        cooper pair box
        deutsch-jozsa algorithm circuit
        deutsch-jozsa quantum circuit diagram
        bernstein-vazirani circuit
        bernstein-vazirani algorithm circuit
        simon algorithm circuit
        quantum query algorithm circuit
        quantum fourier transform circuit
        qft circuit diagram
        shor algorithm circuit
        grover algorithm circuit
        grover search quantum circuit
        quantum counting circuit
        quantum walk circuit
    """,

    "variational_circuits": """
        variational quantum circuit
        parameterized circuit
        ansatz circuit
        vqe circuit
        variational quantum eigensolver
        qaoa circuit
        quantum approximate optimization
        hardware efficient ansatz
        variational quantum classifier
    """,

    "communication_and_entanglement": """
        quantum key distribution circuit
        quantum teleportation circuit
        bell state measurement circuit
        superdense coding circuit
        quantum communication protocol
        epr pair distribution circuit
        entanglement circuit
        entangling gates
    """
}

# ================ TOKEN CONFIGURATIONS ================
PROTECTED_TOKENS = {
    "cnot", "cx", "cz",
    "rx", "ry", "rz",
    "qft", "qaoa", "vqe", "vqc", "vqa",
    "iswap", "ladder"
    # Compound terms from infix normalization
    "multiqubit", "twoqubit", "threequbit", "singlequbit", "nqubit",
    "multigate", "twogate", "threegate",
    "manybody", "twobody", "threebody",
    "controllednot", "controlledz", "controlledx",
    "deutschjozsa", "bernsteinvazirani"
}

QUANTUM_POSITIVE_TOKENS = {
    "quantum", "qubit", "qbit", "gate", "cnot", "hadamard", "pauli",
    "superposition", "entanglement", "coherence", "decoherence",
    "algorithm", "grover", "shor", "qft", "vqe", "qaoa", "ansatz",
    "rx", "ry", "rz", "swap", "iswap", "toffoli", "fredkin",
    "fidelity", "ladder",
    "oracle", "teleportation", "bell", "ghz", "epr",
    "deutsch-jozsa", "bernstein-vazirani", "simon",
    
    "cnot", "cx", "cz",
    "qft", "qaoa", "vqe", "vqc", "vqa",
    "iswap",
}

# Negative tokens for penalty calculation
NEGATIVE_RAW_TOKENS = {
    "equivalent", "simu", "simulation", "simulated", "simul", "theoretical",
    "walk", "grid", "lattice", "lattic", "array", "atom", "molecule", "molecular",
    "atomic", "photon", "spin", "nuclei", "electron", "ionic", "ion",
    "list", "topics", "overview", "summary", "introduction", "conclusion", "refer", "reference", "bibliographi",
    "table", "tabular", "flow", "workflow",
    "axis", "xaxis", "yaxis", "zaxis",
    "label", "title", "legend",
    "plot", "graph", "chart", "histogram",
    "scatter", "bar", "boxplot", "violin",
    "heatmap", "contour", "surface",
    "curve", "trend", "profil", "exampl",
    "implement", "flowchart",
    "demonstr",
    "code", "kernel", "notebook", "script", "function",
    "cuda", "cpu", "gpu", "illustration", "pulse", "duration", "scatter",
    "energy", "level", "spectrum", "eigenvalu","eigenstates" "matrix", "numerics",
    "overlap", "correlation", "concurrence", "log", "coefficient", "covariance",
    "heat", "thermal", "thermodynam", "engin", "temperatur", "entropi",
    "dataset", "benchmark", "simulation", "simul", "iqm", "qpu", "hardware", "outlier",
    "training", "test", "validation", "fold", "cross-valid", "bloch", "sphere", "spherical", "spheric",
    "data", "dyson", "fit", "regress", "classif", "clust", "latice", "lattice", 
    "geometry", "graph", "network", "geometric", "time", "population", "ms","frequency", "domain", "duration", "mod", "modulus",
    "rate", "decay", "decoher", "nois", "signal", "volt", "current", "microsecond", "nanosecond", "millisecond",
    "distribution", "probabl", "expect",
    "varianc", "mean", "averag",
    "standard", "deviat", "confid",
    "interval", "percent", "ratio",
    "sparsity", "histogram", "bin", "sparse",
    "result", "perform", "accuraci",
    "error", "loss", "benchmark",
    "metric", "score", "evaluat",
    "compar", "improv", "gain",
    "energi", "fidel", "fidelities", "overlap",
    "spectrum", "spectra",
    "eigenvalu", "eigenstat",
    "amplitud", "phase",
    "frequenc", "reson",
    "simul", "numer", "comput",
    "trial", "sampl",
    "iteration", "epoch",
    "converg", "optim",
    "measur", "readout",
    "nois", "decoher",
    "calibr", "volt",
    "current", "signal",
    "node", "edge",
    "layout", "topolog",
    "network", "connect",
    "resistor", "capacitor", "inductor", "transistor", "diode", "amplifier",
    "voltage", "current", "frequency", "signal", "impedance", "resistance",
    "capacitance", "inductance", "transmission", "power", "supply",
    "battery", "switch", "relay", "motor", "generator", "transformer",
    "analog", "digital", "pulse", "waveform", "amplitude", "phase",
    "ac", "dc", "alternating", "direct", "oscillator", "filter",
    "opamp", "operational", "mosfet", "bjt", "thyristor", "sensor",
    "3d", "three", "dimensional", "isometric", "perspective",
    "render", "rendering", "visualization", "volume", "mesh",
    "wireframe", "solid", "shaded", "lit", "lighting", "camera",
    "apparatus", "trap", "cavity", "setup", "experiment", "measurem", "detec",
    "atom", "ion", "photon", "spin", "nuclei", "electron",
    "realization", "implement", "experimental",
}

NEGATIVE_TOKENS = {STEMMER.stem(token) for token in NEGATIVE_RAW_TOKENS}

FILENAME_NEGATIVE_RAW = {
    "plot", "graph", "chart", "hist",
    "loss", "acc", "accuracy",
    "result", "results",
    "benchmark", "energy",
    "spectrum", "spectra",
    "prob", "distribution",
    "heatmap", "surface",
    "curve", "spectrum", "distribution",
    "simu", "simulation", "3d", "sphere", "spheric", "spherical",
    "duration", "time",
}

FILENAME_NEGATIVE_TOKENS = {STEMMER.stem(t) for t in FILENAME_NEGATIVE_RAW}