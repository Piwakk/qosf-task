from functools import reduce
from random import choice
from typing import Literal

import numpy as np


I = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
    ],
    dtype=np.float128,
)

X = np.array(
    [
        [0.0, 1.0],
        [1.0, 0.0],
    ],
    dtype=np.float128,
)

H = (
    1
    / np.sqrt(2)
    * np.array(
        [
            [1.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=np.float128,
    )
)

CX = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.float128,
)


_one_qubit_gates = {"I": I, "X": X, "H": H}

_two_qubit_gates = {"CX": CX}


def kron(*args) -> np.ndarray:
    """Return the Kronecker product of its args."""
    return reduce(np.kron, args)


def apply_single_qubit_gate(
    state: np.ndarray,
    gate_name: Literal["I", "X", "H"],
    index: int,
) -> np.ndarray:
    gate = _one_qubit_gates[gate_name]

    # Number of qubits.
    n = int(np.log2(state.shape[0]))

    # Pad with identity gates.
    full_gate = kron(*(I if i != index else gate for i in range(n)))

    return full_gate @ state


def apply_two_qubit_gate(
    state: np.ndarray, gate_name: Literal["CX"], index: tuple[int, int]
) -> np.ndarray:
    gate = _two_qubit_gates[gate_name]

    if index[1] != index[0] + 1:
        raise ValueError("Must be contiguous qubits")

    # Number of qubits.
    n = int(np.log2(state.shape[0]))

    # Pad with identity gates.
    gates = []
    for i in range(n):
        if i not in index:
            gates.append(I)
        elif i == index[0]:
            gates.append(gate)
    full_gate = kron(*gates)

    return full_gate @ state


def apply_gate(
    state: np.ndarray,
    gate_name: Literal["I", "X", "H", "CX"],
    index: int | tuple[int, int],
) -> np.ndarray:
    if gate_name in _one_qubit_gates:
        return apply_single_qubit_gate(state, gate_name, index)
    return apply_two_qubit_gate(state, gate_name, index)
