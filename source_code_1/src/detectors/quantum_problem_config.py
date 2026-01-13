"""
Quantum Problem Classification - Phrases and Reference Texts
Separated configuration for better maintainability
"""

# Explicit phrases for Stage 1 (high-confidence matches)
EXPLICIT_PHRASES = {
    'Grover_Algorithm': [
        # Core algorithm names
        "grover's algorithm",
        "grover algorithm",
        "grover search",
        "grover's search",
        "grovers algorithm",
        
        # Search variants
        "quantum search algorithm",
        "quantum search",
        "unstructured search",
        "unstructured database search",
        "database search algorithm",
        "unsorted search",
        
        # Amplitude amplification family
        "amplitude amplification",
        "amplitude estimation",
        "quantum amplitude amplification",
        "quantum amplitude estimation",
        "oblivious amplitude amplification",
        "fixed-point amplitude amplification",
        
        # Grover variants
        "multi-target grover",
        "partial grover search",
        "grover adaptive search",
        "gas algorithm",
        "grover-based",
        "grover iterate",
        "grover iteration",
        "grover operator",
        
        # Related algorithms
        "quantum counting",
        "quantum counting algorithm",
        "durr-hoyer algorithm",
        "durr hoyer",
        "minimum finding algorithm",
        "quantum minimum finding",
        "quantum maximum finding",
        "boyer-brassard-hoyer-tapp",
        "bbht algorithm",
        
        # Key components
        "grover diffusion",
        "diffusion operator",
        "inversion about mean",
        "inversion about the mean",
        "grover oracle",
        "oracle operator",
        "phase oracle",
        "marking oracle",
        "sign oracle",
        
        # Technical terms
        "quadratic speedup search",
        "sqrt(n) query",
        "o(sqrt(n))",
        "optimal quantum search"
    ],
    
    'Shor_Algorithm': [
        # Core algorithm names
        "shor's algorithm",
        "shor algorithm",
        "shors algorithm",
        "shor's factoring",
        "shor factoring",
        
        # Factorization
        "quantum factorization",
        "quantum factoring",
        "integer factorization algorithm",
        "integer factorization quantum",
        "prime factorization quantum",
        "factoring algorithm",
        "number factoring",
        "semiprime factoring",
        
        # Period/Order finding
        "period finding",
        "period-finding",
        "quantum period finding",
        "order finding",
        "order-finding algorithm",
        "order finding quantum",
        
        # Discrete logarithm
        "discrete logarithm",
        "discrete log quantum",
        "quantum discrete logarithm",
        "dlp quantum",
        "ecdlp quantum",
        "elliptic curve discrete logarithm",
        
        # Cryptographic implications
        "rsa breaking",
        "break rsa",
        "cryptanalysis quantum",
        "post-quantum threat",
        "quantum threat to rsa",
        
        # Key components
        "modular exponentiation",
        "modular arithmetic quantum",
        "controlled modular multiplication",
        "continued fractions",
        "continued fraction expansion",
        
        # Circuit variants
        "beauregard circuit",
        "2n+3 qubits",
        "kitaev's approach",
        "semiclassical shor"
    ],
    
    'QFT_QPE': [
        # QFT terms
        "quantum fourier transform",
        "qft circuit",
        "qft algorithm",
        "inverse qft",
        "inverse quantum fourier transform",
        "iqft",
        "approximate qft",
        "semi-classical qft",
        "semiclassical qft",
        
        # QPE terms
        "quantum phase estimation",
        "phase estimation algorithm",
        "qpe algorithm",
        "qpe circuit",
        "iterative phase estimation",
        "iterative qpe",
        "bayesian phase estimation",
        "robust phase estimation",
        
        # Eigenvalue estimation
        "eigenvalue estimation",
        "quantum eigenvalue estimation",
        "eigenphase estimation",
        "qeep",
        
        # Key concepts
        "phase kickback",
        "phase kick-back",
        "phase register",
        "precision qubits",
        "hadamard ladder",
        "controlled rotation ladder",
        "rotation ladder circuit",
        
        # Technical terms
        "cr_k gate",
        "controlled phase rotation",
        "2pi/2^k rotation",
        "twiddle factor",
        "bit reversal",
        "fourier basis",
        "frequency domain quantum"
    ],
    
    'VQE': [
        # Core algorithm names
        "variational quantum eigensolver",
        "vqe algorithm",
        "vqe method",
        "vqe circuit",
        "variational eigensolver",
        
        # Variants
        "adapt-vqe",
        "adapt vqe",
        "qubit-adapt-vqe",
        "qeb-adapt-vqe",
        "orbital-optimized vqe",
        "oo-vqe",
        "ssvqe",
        "subspace-search vqe",
        "vqd",
        "variational quantum deflation",
        "cluster vqe",
        "multireference vqe",
        
        # Ansatz types
        "uccsd ansatz",
        "uccsd circuit",
        "unitary coupled cluster",
        "coupled cluster singles doubles",
        "hardware-efficient ansatz",
        "hardware efficient ansatz",
        "hea ansatz",
        "chemistry-inspired ansatz",
        "problem-inspired ansatz",
        
        # Quantum chemistry
        "molecular ground state",
        "ground state energy",
        "molecular hamiltonian",
        "electronic structure",
        "quantum chemistry vqe",
        "molecular simulation vqe",
        "chemical accuracy",
        
        # Key concepts
        "variational principle",
        "parameterized ansatz",
        "classical optimizer vqe",
        "hybrid quantum-classical optimization",
        "expectation value measurement",
        "hamiltonian averaging",
        
        # Mapping terms
        "jordan-wigner vqe",
        "bravyi-kitaev vqe",
        "fermionic mapping",
        "qubit mapping chemistry"
    ],
    
    'QAOA': [
        # Core algorithm names
        "quantum approximate optimization algorithm",
        "qaoa",
        "qaoa algorithm",
        "qaoa circuit",
        "quantum alternating operator ansatz",
        
        # Variants
        "rqaoa",
        "recursive qaoa",
        "warm-start qaoa",
        "warm start qaoa",
        "multi-angle qaoa",
        "ma-qaoa",
        "grover-mixer qaoa",
        "xy-mixer qaoa",
        "xqaoa",
        "qaoa+",
        "adaptive qaoa",
        "layer-wise qaoa",
        "parameter-setting qaoa",
        
        # Key components
        "cost hamiltonian",
        "problem hamiltonian",
        "mixer hamiltonian",
        "driver hamiltonian",
        "phase separator",
        "cost layer",
        "mixer layer",
        "p-layer qaoa",
        "p layers qaoa",
        
        # Parameters
        "beta gamma parameters",
        "qaoa parameters",
        "qaoa angles",
        "variational parameters qaoa",
        
        # Related terms
        "alternating operator",
        "alternating layers",
        "quantum optimization variational",
        "approximate optimization",
        "combinatorial qaoa",
        "maxcut qaoa",
        "constraint qaoa"
    ],
    
    'Quantum_Simulation': [
        # Core terms
        "quantum simulation",
        "hamiltonian simulation",
        "quantum hamiltonian simulation",
        "digital quantum simulation",
        "analog quantum simulation",
        
        # Time evolution
        "time evolution simulation",
        "time evolution operator",
        "unitary time evolution",
        "e^(-iht) simulation",
        "propagator simulation",
        "real-time evolution",
        "imaginary-time evolution",
        
        # Trotterization
        "trotterization",
        "trotter decomposition",
        "trotter-suzuki",
        "trotter suzuki decomposition",
        "product formula",
        "lie-trotter",
        "lie trotter formula",
        "first-order trotter",
        "second-order trotter",
        "higher-order trotter",
        "trotter steps",
        "trotter layers",
        "trotter error",
        
        # Advanced methods
        "linear combination of unitaries",
        "lcu method",
        "qubitization",
        "quantum signal processing",
        "qsp simulation",
        "qsvt",
        "quantum singular value transformation",
        "block encoding",
        
        # Physical models
        "fermi-hubbard simulation",
        "hubbard model simulation",
        "heisenberg model simulation",
        "spin chain simulation",
        "lattice gauge theory",
        "molecular dynamics simulation",
        "many-body simulation",
        "condensed matter simulation",
        
        # Chemistry simulation
        "quantum chemistry simulation",
        "molecular simulation quantum",
        "electronic structure simulation",
        "chemical simulation",
        
        # Variational simulation
        "variational quantum simulation",
        "vqs",
        "adiabatic state preparation",
        "ground state preparation"
    ],
    
    'Quantum_Machine_Learning': [
        # Core terms
        "quantum machine learning",
        "qml",
        "quantum ml",
        
        # Neural networks
        "quantum neural network",
        "qnn",
        "quantum deep learning",
        "quantum perceptron",
        "quantum convolutional neural network",
        "qcnn",
        "quantum recurrent neural network",
        "qrnn",
        
        # Classifiers
        "quantum classifier",
        "quantum classification",
        "variational quantum classifier",
        "vqc",
        "quantum binary classifier",
        "quantum multiclass classifier",
        
        # Kernel methods
        "quantum kernel",
        "quantum kernel estimation",
        "quantum kernel method",
        "quantum svm",
        "qsvm",
        "quantum support vector machine",
        "quantum feature map",
        "quantum embedding",
        
        # Other QML algorithms
        "quantum pca",
        "quantum principal component analysis",
        "quantum dimensionality reduction",
        "quantum clustering",
        "quantum autoencoder",
        "quantum generative adversarial",
        "quantum gan",
        "qgan",
        "quantum boltzmann machine",
        "quantum reservoir computing",
        "quantum reinforcement learning",
        
        # Data encoding
        "amplitude encoding",
        "angle encoding",
        "basis encoding",
        "dense encoding",
        "data encoding quantum",
        "data re-uploading",
        "iqp encoding",
        
        # Technical terms
        "parameterized quantum circuit ml",
        "barren plateau",
        "trainability quantum",
        "expressibility",
        "entangling capability",
        "parameter-shift rule",
        "quantum gradient",
        
        # HHL and linear algebra
        "hhl algorithm",
        "quantum linear systems",
        "quantum matrix inversion"
    ],
    
    'Quantum_Cryptography': [
        # Core terms
        "quantum cryptography",
        "quantum key distribution",
        "qkd",
        
        # BB84 family
        "bb84 protocol",
        "bb84",
        "bennett brassard 84",
        "prepare-and-measure qkd",
        "decoy state bb84",
        "decoy-state protocol",
        
        # E91 and entanglement-based
        "e91 protocol",
        "e91",
        "ekert protocol",
        "entanglement-based qkd",
        "entanglement based qkd",
        
        # Other protocols
        "b92 protocol",
        "b92",
        "bbm92",
        "six-state protocol",
        "six state protocol",
        "sarg04",
        "coherent one-way",
        "cow protocol",
        "differential phase shift",
        "dps-qkd",
        
        # Advanced QKD
        "device-independent qkd",
        "diqkd",
        "di-qkd",
        "measurement-device-independent",
        "mdi-qkd",
        "twin-field qkd",
        "tf-qkd",
        "continuous-variable qkd",
        "cv-qkd",
        
        # Security terms
        "quantum secure communication",
        "unconditional security",
        "information-theoretic security",
        "eavesdropping detection",
        "eve attack",
        "intercept-resend attack",
        "photon number splitting",
        "pns attack",
        
        # Key processing
        "key sifting",
        "privacy amplification",
        "error correction qkd",
        "qber",
        "quantum bit error rate",
        "secret key rate",
        
        # Post-quantum
        "post-quantum cryptography",
        "quantum-resistant",
        "quantum-safe",
        "lattice-based cryptography",
        
        # Other quantum crypto
        "quantum digital signature",
        "quantum secret sharing",
        "quantum coin flipping",
        "quantum bit commitment",
        "quantum money",
        "quantum authentication"
    ],
    
    'Error_Correction': [
        # Core terms
        "quantum error correction",
        "qec",
        "quantum error correcting code",
        "error correction code",
        
        # Stabilizer codes
        "stabilizer code",
        "stabilizer formalism",
        "stabilizer measurement",
        "pauli stabilizer",
        "css code",
        "calderbank-shor-steane",
        
        # Specific codes
        "surface code",
        "rotated surface code",
        "planar code",
        "toric code",
        "color code",
        "steane code",
        "shor code",
        "5-qubit code",
        "five qubit code",
        "perfect code",
        "bacon-shor code",
        "subsystem code",
        "repetition code",
        "bit flip code",
        "phase flip code",
        
        # LDPC codes
        "ldpc code",
        "low-density parity-check",
        "quantum ldpc",
        "qldpc",
        
        # Bosonic codes
        "gkp code",
        "gottesman-kitaev-preskill",
        "cat code",
        "cat qubit",
        "binomial code",
        "bosonic code",
        
        # Topological codes
        "topological code",
        "topological quantum error correction",
        "anyonic code",
        
        # Fault tolerance
        "fault-tolerant",
        "fault tolerant quantum computing",
        "ftqc",
        "threshold theorem",
        "error threshold",
        "fault-tolerant gate",
        "transversal gate",
        "magic state distillation",
        "magic state injection",
        "lattice surgery",
        "code switching",
        
        # Syndrome
        "syndrome measurement",
        "syndrome extraction",
        "syndrome decoding",
        "error syndrome",
        "parity check",
        "ancilla measurement",
        
        # Decoding
        "decoder",
        "minimum weight perfect matching",
        "mwpm decoder",
        "union-find decoder",
        "neural network decoder",
        "bp-osd",
        "belief propagation",
        
        # Error mitigation (related)
        "error mitigation",
        "zero-noise extrapolation",
        "zne",
        "probabilistic error cancellation",
        "pec",
        "clifford data regression",
        "cdr",
        "readout error mitigation",
        "measurement error mitigation",
        
        # Logical operations
        "logical qubit",
        "logical gate",
        "encoded qubit",
        "code distance",
        "[[n,k,d]] code"
    ],
    
    'Hardware_Architecture': [
        # Superconducting
        "superconducting qubit",
        "superconducting quantum",
        "transmon qubit",
        "transmon",
        "flux qubit",
        "fluxonium",
        "xmon qubit",
        "charge qubit",
        "phase qubit",
        "josephson junction",
        "squid",
        "coplanar waveguide",
        "cpw resonator",
        
        # Ion traps
        "ion trap",
        "trapped ion",
        "trapped-ion quantum",
        "ion trap quantum",
        "paul trap",
        "penning trap",
        "linear ion trap",
        "surface ion trap",
        "qccd architecture",
        "ion shuttling",
        "hyperfine qubit",
        "optical qubit ion",
        "molmer-sorensen gate",
        
        # Photonic
        "photonic quantum",
        "photonic qubit",
        "linear optical quantum",
        "loqc",
        "boson sampling",
        "gaussian boson sampling",
        "photonic chip",
        "integrated photonics",
        "single photon",
        "squeezed light",
        "knill-laflamme-milburn",
        "klm protocol",
        
        # Neutral atoms
        "neutral atom",
        "neutral-atom quantum",
        "rydberg atom",
        "rydberg qubit",
        "optical tweezer",
        "tweezer array",
        "atom array",
        "rydberg blockade",
        "rydberg interaction",
        
        # Topological
        "topological qubit",
        "majorana qubit",
        "majorana fermion",
        "majorana zero mode",
        "non-abelian anyon",
        "topological quantum computing",
        
        # Spin qubits
        "spin qubit",
        "quantum dot qubit",
        "silicon spin qubit",
        "si/sige quantum dot",
        "nv center",
        "nitrogen-vacancy",
        "diamond nv",
        "electron spin qubit",
        "nuclear spin qubit",
        
        # General hardware
        "quantum processor",
        "quantum chip",
        "quantum hardware",
        "qubit implementation",
        "qubit design",
        "qubit architecture",
        "quantum device",
        "quantum computer hardware",
        
        # Performance metrics
        "gate fidelity",
        "single-qubit fidelity",
        "two-qubit fidelity",
        "coherence time",
        "t1 time",
        "t2 time",
        "t2*",
        "t2 echo",
        "decoherence",
        "relaxation time",
        "dephasing time",
        
        # Connectivity
        "qubit connectivity",
        "coupling map",
        "qubit topology",
        "heavy-hex topology",
        "square lattice topology",
        "all-to-all connectivity",
        "nearest-neighbor",
        
        # Control
        "microwave control",
        "pulse control",
        "gate calibration",
        "qubit control",
        "cryogenic",
        "dilution refrigerator",
        "millikelvin",
        
        # Native gates
        "native gate set",
        "basis gates",
        "sx gate",
        "ecr gate",
        "iswap gate",
        "sqrt-iswap",
        "cz gate native",
        "cross-resonance"
    ],
    
    'Benchmarking': [
        # Supremacy/Advantage
        "quantum supremacy",
        "quantum advantage",
        "quantum computational advantage",
        "beyond-classical",
        "classical intractability",
        
        # Quantum Volume
        "quantum volume",
        "qv metric",
        "heavy output probability",
        "heavy output generation",
        "square circuit",
        "model circuit",
        
        # Randomized Benchmarking
        "randomized benchmarking",
        "rb protocol",
        "clifford randomized benchmarking",
        "interleaved rb",
        "interleaved randomized benchmarking",
        "irb",
        "simultaneous rb",
        "direct rb",
        "cycle benchmarking",
        "mirror rb",
        "mirror circuits",
        "character benchmarking",
        
        # Cross-entropy
        "cross-entropy benchmarking",
        "xeb",
        "linear xeb",
        "cross entropy difference",
        "f_xeb",
        "random circuit sampling",
        "rcs",
        
        # Fidelity estimation
        "fidelity estimation",
        "gate fidelity estimation",
        "process fidelity",
        "state fidelity",
        "average gate fidelity",
        "error per clifford",
        "epc",
        "error per gate",
        "epg",
        "infidelity",
        
        # Tomography
        "gate set tomography",
        "gst",
        "quantum process tomography",
        "qpt",
        "quantum state tomography",
        "qst",
        "spam errors",
        "spam characterization",
        
        # System metrics
        "clops",
        "circuit layer operations per second",
        "layer fidelity",
        "algorithmic qubits",
        "eplg",
        "error per layered gate",
        
        # Noise characterization
        "noise characterization",
        "noise benchmarking",
        "crosstalk characterization",
        "leakage benchmarking",
        "t1 measurement",
        "t2 measurement",
        "coherence benchmarking",
        
        # Performance testing
        "quantum benchmark",
        "benchmarking protocol",
        "performance metric",
        "quantum computer benchmark",
        "scalability benchmark",
        "fidelity benchmark"
    ],
    
    'Optimization_Problems': [
        # Core terms
        "quantum optimization",
        "combinatorial optimization quantum",
        "discrete optimization quantum",
        
        # QUBO/Ising
        "qubo",
        "quadratic unconstrained binary optimization",
        "ising model optimization",
        "ising hamiltonian optimization",
        "spin glass optimization",
        "binary optimization quantum",
        
        # MaxCut
        "maxcut problem",
        "maxcut quantum",
        "max-cut",
        "max cut problem",
        "maximum cut",
        "graph cut problem",
        "weighted maxcut",
        
        # Graph problems
        "graph coloring quantum",
        "vertex coloring",
        "chromatic number",
        "vertex cover quantum",
        "minimum vertex cover",
        "independent set quantum",
        "maximum independent set",
        "mis problem",
        "clique problem quantum",
        "maximum clique",
        "graph partitioning",
        "community detection quantum",
        
        # TSP and routing
        "traveling salesman quantum",
        "tsp quantum",
        "travelling salesman",
        "vehicle routing quantum",
        "vrp quantum",
        "shortest path quantum",
        "hamiltonian path",
        
        # Scheduling
        "job shop scheduling quantum",
        "scheduling optimization",
        "resource allocation quantum",
        "task scheduling",
        "flow shop scheduling",
        
        # Satisfiability
        "max-sat quantum",
        "maxsat",
        "satisfiability quantum",
        "sat problem quantum",
        "constraint satisfaction quantum",
        "csp quantum",
        "boolean satisfiability",
        
        # Knapsack
        "knapsack problem quantum",
        "subset sum quantum",
        "bin packing quantum",
        "number partitioning quantum",
        
        # Financial
        "portfolio optimization quantum",
        "financial optimization quantum",
        "asset allocation quantum",
        "risk optimization",
        "mean-variance optimization",
        
        # Other optimization
        "integer programming quantum",
        "linear programming quantum",
        "quadratic programming",
        "mixed-integer programming",
        "constrained optimization quantum",
        
        # Technical terms
        "penalty function",
        "penalty term",
        "constraint encoding",
        "slack variable",
        "lagrange multiplier quantum",
        "feasibility",
        "objective function quantum",
        "cost function optimization",
        
        # Algorithms
        "quantum annealing",
        "adiabatic optimization",
        "variational optimization",
        "grover optimization"
    ]
}

# Reference texts for Stage 2 (SciBERT similarity)
REFERENCE_TEXTS = {
    'Grover_Algorithm': """
        Grover's algorithm quantum search unstructured database amplitude amplification 
        quadratic speedup O(sqrt(N)) oracle diffusion operator inversion about mean 
        phase oracle marking oracle Grover iterate Grover diffusion multi-target search 
        partial search quantum counting amplitude estimation Durr-Hoyer minimum finding 
        Boyer-Brassard-Hoyer-Tapp BBHT fixed-point amplitude amplification Grover adaptive 
        search GAS oracle query black-box function marked element target state search space 
        Hadamard superposition multi-controlled gates MCX MCZ Toffoli CCX reflection operator 
        2|s><s|-I optimal iterations pi/4 sqrt(N) database search unsorted search 
        quantum speedup query complexity boolean oracle sign oracle bit-flip oracle
    """,
    
    'Shor_Algorithm': """
        Shor's algorithm integer factorization polynomial time quantum Fourier transform 
        QFT period finding order finding modular exponentiation modular arithmetic 
        continued fractions RSA cryptography prime factorization discrete logarithm 
        DLP ECDLP elliptic curve post-quantum cryptography exponential speedup 
        Beauregard circuit 2n+3 qubits control register work register controlled-U 
        controlled multiplication modular addition phase estimation eigenvalue 
        coprime GCD Euclidean algorithm Euler totient semiprime factoring 
        number theory modular inverse period r order r a^x mod N classical 
        post-processing convergents semiclassical implementation Kitaev approach
    """,
    
    'QFT_QPE': """
        quantum Fourier transform QFT inverse QFT IQFT discrete Fourier transform 
        quantum phase estimation QPE eigenvalue estimation phase kickback 
        controlled rotation CR CRz CRk phase gate R_k rotation 2pi/2^k 
        Hadamard ladder staircase pattern twiddle factors product state 
        binary fraction 0.j1j2j3 precision qubits ancilla qubits eigenstate 
        eigenvector eigenphase controlled-U CU gates Hadamard test 
        iterative phase estimation Kitaev algorithm Bayesian phase estimation 
        robust phase estimation QEEP quantum eigenvalue estimation 
        approximate QFT semi-classical QFT SWAP network bit reversal 
        frequency domain spectral analysis unitary eigenvalues
    """,
    
    'VQE': """
        variational quantum eigensolver VQE hybrid quantum-classical algorithm 
        ground state energy molecular Hamiltonian expectation value variational 
        principle ansatz parameterized quantum circuit PQC UCCSD unitary coupled 
        cluster singles doubles hardware-efficient ansatz HEA ADAPT-VQE qubit-ADAPT 
        QEB-ADAPT orbital optimization classical optimizer COBYLA SPSA L-BFGS-B 
        Adam gradient descent parameter optimization cost function loss function 
        barren plateau vanishing gradient quantum chemistry electronic structure 
        molecular orbital active space Hartree-Fock reference state chemical accuracy 
        correlation energy Jordan-Wigner transformation Bravyi-Kitaev parity mapping 
        fermionic operators second quantization creation annihilation operators 
        Pauli strings measurement excited states SSVQE VQD quantum subspace expansion
    """,
    
    'QAOA': """
        quantum approximate optimization algorithm QAOA variational quantum algorithm 
        combinatorial optimization cost Hamiltonian mixer Hamiltonian driver Hamiltonian 
        phase separator alternating layers p-layers beta gamma variational parameters 
        MaxCut Max-SAT graph problems NP-hard approximation ratio QUBO Ising model 
        ZZ interaction cost layer mixer layer RX mixer transverse field initial 
        superposition |+> state RQAOA recursive QAOA warm-start QAOA multi-angle QAOA 
        Grover mixer XY mixer XQAOA quantum alternating operator ansatz Hadfield 
        adiabatic limit graph encoding edge weights constraint satisfaction 
        combinatorial problems integer programming quadratic optimization 
        optimization landscape parameter transfer success probability
    """,
    
    'Quantum_Simulation': """
        quantum simulation Hamiltonian simulation time evolution e^(-iHt) propagator 
        Trotter decomposition Trotterization Trotter-Suzuki product formula Trotter 
        steps Trotter layers Trotter error Lie-Trotter first-order second-order 
        symmetric Suzuki many-body physics condensed matter molecular dynamics 
        quantum chemistry Pauli strings Pauli decomposition exponential of Pauli 
        Jordan-Wigner Bravyi-Kitaev fermionic simulation Fermi-Hubbard model 
        Heisenberg model Ising model spin chain lattice model time-dependent 
        adiabatic evolution linear combination of unitaries LCU qubitization 
        quantum signal processing QSP QSVT block encoding PREP SEL variational 
        quantum simulation digital quantum simulation analog simulation
    """,
    
    'Quantum_Machine_Learning': """
        quantum machine learning QML quantum neural network QNN variational quantum 
        circuit VQC parameterized quantum circuit PQC quantum classifier classification 
        feature map quantum embedding data encoding amplitude encoding angle encoding 
        basis encoding dense encoding IQP encoding data re-uploading quantum kernel 
        kernel estimation fidelity kernel quantum SVM QSVM quantum PCA dimensionality 
        reduction quantum convolutional neural network QCNN quantum GAN qGAN quantum 
        autoencoder quantum Boltzmann machine HHL algorithm linear systems barren 
        plateau trainability expressibility entangling capability parameter-shift rule 
        gradient estimation hybrid classical-quantum training landscape quantum 
        advantage pattern recognition quantum reservoir computing reinforcement learning
    """,
    
    'Quantum_Cryptography': """
        quantum cryptography quantum key distribution QKD BB84 protocol E91 protocol 
        B92 BBM92 six-state protocol decoy state MDI-QKD measurement device independent 
        device independent DIQKD continuous variable CV-QKD twin-field TF-QKD 
        photon polarization basis choice rectilinear basis diagonal basis complementary 
        bases mutually unbiased bases sifting error correction privacy amplification 
        QBER quantum bit error rate secret key rate eavesdropping Eve intercept-resend 
        attack no-cloning theorem Bell inequality CHSH inequality Bell states EPR pairs 
        entanglement-based quantum digital signature quantum secret sharing post-quantum 
        cryptography secure communication information-theoretic security key exchange
    """,
    
    'Error_Correction': """
        quantum error correction QEC stabilizer code stabilizer formalism syndrome 
        measurement syndrome extraction logical qubit physical qubit ancilla qubit 
        code distance [[n,k,d]] notation fault-tolerant fault tolerance threshold 
        theorem error threshold surface code rotated surface code color code toric 
        code Steane code Shor code CSS code repetition code LDPC code Bacon-Shor 
        subsystem code GKP code cat code bosonic code transversal gates magic state 
        distillation lattice surgery flag qubits decoder minimum weight perfect 
        matching MWPM union-find error mitigation zero-noise extrapolation ZNE 
        probabilistic error cancellation PEC Clifford data regression readout error 
        bit flip phase flip depolarizing noise decoherence T1 T2 coherence
    """,
    
    'Hardware_Architecture': """
        quantum hardware quantum processor qubit implementation superconducting qubit 
        transmon flux qubit fluxonium Josephson junction trapped ion ion trap 
        hyperfine qubit optical qubit QCCD architecture photonic qubit linear optical 
        neutral atom Rydberg atom optical tweezer topological qubit Majorana fermion 
        spin qubit quantum dot NV center gate fidelity single-qubit fidelity two-qubit 
        fidelity coherence time T1 relaxation T2 dephasing T2* T2echo crosstalk 
        leakage qubit connectivity coupling map topology heavy-hex square lattice 
        all-to-all native gates basis gates SX RZ CX CZ ECR iSWAP Molmer-Sorensen 
        dilution refrigerator cryogenic millikelvin microwave control calibration 
        SWAP network qubit routing transpilation circuit compilation
    """,
    
    'Benchmarking': """
        quantum benchmarking randomized benchmarking RB interleaved RB IRB Clifford 
        gates Clifford group error per Clifford EPC error per gate EPG gate fidelity 
        process fidelity state fidelity infidelity depolarizing parameter quantum 
        volume QV heavy output probability square circuit cross-entropy benchmarking 
        XEB linear XEB F_XEB quantum supremacy quantum advantage random circuit 
        sampling gate set tomography GST quantum process tomography quantum state 
        tomography SPAM errors cycle benchmarking mirror circuits direct RB 
        simultaneous RB CLOPS circuit layer operations layer fidelity algorithmic 
        qubits fidelity estimation performance metrics noise characterization 
        system benchmarking scalability assessment quantum computer evaluation
    """,
    
    'Optimization_Problems': """
        quantum optimization combinatorial optimization QUBO quadratic unconstrained 
        binary optimization Ising model Ising Hamiltonian spin glass spin variables 
        objective function cost function ground state energy penalty terms penalty 
        coefficients slack variables constraints Lagrange multipliers MaxCut maximum 
        cut graph coloring vertex cover independent set traveling salesman TSP 
        vehicle routing VRP job shop scheduling knapsack problem Max-SAT satisfiability 
        number partitioning portfolio optimization financial optimization constraint 
        satisfaction CSP graph problems NP-hard NP-complete binary variables 
        quadratic terms linear terms ZZ coupling Z operator problem Hamiltonian 
        solution encoding feasibility infeasibility optimization landscape
    """
}
