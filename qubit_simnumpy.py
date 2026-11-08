"""
A small qubit simulator using NumPy"""

import numpy as np

#basic states
ket0 = np.array([1, 0], dtype=complex)   #|0>
ket1 = np.array([0, 1], dtype=complex)   #|1>

#one-qubit gates
X = np.array([[0, 1],
              [1, 0]], dtype=complex)                 #Pauli X (bit flip)

Z = np.array([[1, 0],
              [0, -1]], dtype=complex)                #Pauli Z (phase flip)

H = np.array([[1, 1],
              [1, -1]], dtype=complex) / np.sqrt(2)   #Hadamard

S = np.array([[1, 0],
              [0, 1j]], dtype=complex)                #S gate: square root of Z

SQRT_X = 0.5 * np.array([[1 + 1j, 1 - 1j],
                         [1 - 1j, 1 + 1j]], dtype=complex)   #square root of NOT

I = np.eye(2, dtype=complex)                          #identity (does nothing)

# The same square root of NOT, built a second way: H S H
# H swaps the roles of X and Z, so "square root of Z" (S) becomes "square root of X".
HSH = H @ S @ H

#two-qubit gate
# Control is the first qubit, target is the second.
# Basis order: |00>, |01>, |10>, |11>(lexicographically). If the control is 1, flip the target.
CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]], dtype=complex)


#functions for applications
def apply(gate, state):
    """Apply a gate (matrix) to a state (vector)."""
    return gate @ state


def two_qubit_state(q_first, q_second):
    """Combine two single-qubit states into one 2-qubit state (tensor product)."""
    return np.kron(q_first, q_second)


def probabilities(state):
    """Probability of each measurement outcome = |amplitude|^2."""
    return np.abs(state) ** 2


def show(label, state):
    """Print the probability of each outcome, e.g. |00>, |01>, ..."""
    n_qubits = int(np.log2(len(state)))
    probs = probabilities(state)
    parts = [f"|{i:0{n_qubits}b}>: {p:.3f}" for i, p in enumerate(probs)]
    print(f"{label:38s} " + "   ".join(parts))


def is_unitary(gate):
    """A valid quantum gate satisfies U^dagger U = I."""
    return np.allclose(gate.conj().T @ gate, np.eye(len(gate)))


def is_entangled(state):
    """
    Check whether a 2-qubit state is entangled.
    Write the 4 amplitudes as a 2x2 table. If the table has rank 1, the state
    is a product of two single-qubit states. If it has rank 2, it is entangled.
    """
    table = state.reshape(2, 2)
    return np.linalg.matrix_rank(table, tol=1e-10) == 2


if __name__ == "__main__":
    print("Check that every gate is unitary:")
    for name, g in [("X", X), ("Z", Z), ("H", H), ("S", S), ("sqrt(X)", SQRT_X), ("CNOT", CNOT)]:
        print(f"  {name:8s} unitary: {is_unitary(g)}")

    print("\nSingle qubit examples:")
    show("|0>", ket0)
    show("X|0>  (bit flip -> |1>)", apply(X, ket0))
    show("H|0>  (equal superposition)", apply(H, ket0))
    show("H H |0>  (back to |0>)", apply(H, apply(H, ket0)))

    plus = apply(H, ket0)                      #|+> = (|0> + |1>)/sqrt(2)
    minus = apply(Z, plus)                 
    print("\nPhase flip: Z changes the sign, but not the probabilities")
    print("|+> amplitudes :", np.round(plus, 3))
    print("Z|+> amplitudes:", np.round(minus, 3))
    show("Z|+> probabilities", minus)
    show("H Z|+>  (H turns the phase into a bit)", apply(H, minus))

    print("\nSquare root of NOT, two ways:")
    half = apply(SQRT_X, ket0)
    show("sqrt(X)|0>  (halfway to |1>)", half)
    show("sqrt(X) sqrt(X)|0>  (= X|0> = |1>)", apply(SQRT_X, half))
    print("sqrt(X) @ sqrt(X) equals X      :", np.allclose(SQRT_X @ SQRT_X, X))
    print("H S H equals sqrt(X)            :", np.allclose(HSH, SQRT_X))
    print("S @ S equals Z (S is sqrt of Z) :", np.allclose(S @ S, Z))
    print("H S H matrix:")
    print(np.round(HSH, 3))
    show("H S H |0>  (same result as sqrt(X))", apply(HSH, ket0))

    print("\nTwo-qubit states explained")
    print("  |ab> means: first qubit is a, second qubit is b.")
    print("  The 4 amplitudes are ordered |00>, |01>, |10>, |11>,")
    print("  so the position in the vector is the binary number ab.")
    for a, qa in [(0, ket0), (1, ket1)]:
        for b, qb in [(0, ket0), (1, ket1)]:
            print(f"    |{a}{b}> = kron(|{a}>, |{b}>) = {np.real(two_qubit_state(qa, qb)).astype(int)}")

    print("\n  A product state: first qubit in |+>, second in |0>")
    prod = two_qubit_state(plus, ket0)
    print("    amplitudes:", np.round(prod, 3))
    show("    probabilities", prod)
    print("    entangled?", is_entangled(prod), " (each qubit still has its own state)")

    print("\n  Gates on one qubit of a pair:")
    print("    H on the first qubit : kron(H, I)")
    print("    H on the second qubit: kron(I, H)")
    s = two_qubit_state(ket0, ket0)            # |00>
    show("    kron(I, H)|00>", apply(np.kron(I, H), s))

    print("\n  CNOT flips the second qubit only if the first is 1:")
    show("    CNOT |10>  (control 1 -> flips)", apply(CNOT, two_qubit_state(ket1, ket0)))
    show("    CNOT |00>  (control 0 -> nothing)", apply(CNOT, s))

    print("\nBell state: H on first qubit, then CNOT")
    after_h = apply(np.kron(H, I), s)          # H on first qubit only
    bell = apply(CNOT, after_h)
    print("  after H  :", np.round(after_h, 3), " entangled?", is_entangled(after_h))
    print("  after CNOT:", np.round(bell, 3), " entangled?", is_entangled(bell))
    show("  Bell state probabilities", bell)
    print("  Only |00> and |11> ever appear: the two qubits always agree.")
    print("  No pair of single-qubit states multiplies to give this, so it is entangled.")