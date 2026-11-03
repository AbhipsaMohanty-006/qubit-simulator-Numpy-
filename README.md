# qubit-simulator-Numpy-
Simulation of single and two-qubit states and basic quantum gates (X, Z, H, CNOT) using NumPy.
# Qubit Simulator in NumPy

A small quantum simulator written with NumPy only, with no quantum libraries. A qubit is a 2-element vector and each gate is a matrix, so applying a gate is a matrix multiplication. I wrote this to understand what happens underneath when Qiskit applies a gate.

# What it includes
- Single-qubit states |0> and |1>
- Gates: Pauli X (bit flip), Pauli Z (phase flip), Hadamard (H), S (Phase operation of pi/2 flip) , square root of NOT, and CNOT
- The square root of NOT built two ways, and a check that they agree: written directly as a matrix, and as H · S · H
- Two-qubit states built with the tensor product (`np.kron`)
- Measurement probabilities from squared amplitudes
- A check that every gate is unitary
- A check for whether a two-qubit state is entangled
- A Bell state built from H and CNOT

# Square root of NOT, two ways
The square root of NOT is a gate that gives NOT (X) when applied twice. Applying it once puts a qubit halfway between |0> and |1>.

- Directly: `0.5 * [[1+i, 1-i], [1-i, 1+i]]`
- As H S H: S is the square root of Z (it adds a phase of i to |1>). H swaps the roles of X and Z, so sandwiching S between two H gates turns "square root of Z" into "square root of X". The code checks that `H @ S @ H` equals the direct matrix.

# How two-qubit states work
- `|ab>` means the first qubit is in state a and the second is in state b.
- A two-qubit state has 4 amplitudes, in the order |00>, |01>, |10>, |11>. The position in the vector is the binary number ab, so |01> is `[0, 1, 0, 0]`.
- Two separate qubits are combined with the tensor product: `np.kron(first, second)`.
- A gate on one qubit of a pair is combined with the identity: `kron(H, I)` is H on the first qubit and `kron(I, H)` is H on the second.
- CNOT flips the second qubit only when the first qubit is 1.
- Product vs entangled: a state like |+>|0> can be split back into one state per qubit. The Bell state cannot, which is what entangled means. The code tests this by writing the 4 amplitudes as a 2x2 table: rank 1 means product, rank 2 means entangled.
- Qubit order: this simulator writes the first qubit on the left. Qiskit writes qubit 0 on the right, so bit strings can look reversed when you compare the two.
