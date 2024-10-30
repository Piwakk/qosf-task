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
).reshape(2, 2, 2, 2)

_one_qubit_gates = {"I": I, "X": X, "H": H}

_two_qubit_gates = {"CX": CX}


def kron(*args) -> np.ndarray:
    """Return the Kronecker product of its args."""
    return reduce(lambda x, y: np.tensordot(x, y, axes=0), args)


def apply_single_qubit_gate(
    state: np.ndarray,
    gate_name: Literal["I", "X", "H"],
    index: int,
) -> np.ndarray:
    # Example:
    #   - n = 3
    #   - index = 1
    #
    # The subscripts of `np.einsum()` are
    #   - `state`: [0, 1, 2]
    #   - `gate`: [3, 1]
    #   - output: [0, 3, 2]

    gate = _one_qubit_gates[gate_name]
    n = state.ndim

    return np.einsum(
        state,
        [x for x in range(n)],
        gate,
        [n, index],
        [n if x == index else x for x in range(n)],
    )


def apply_two_qubit_gate(
    state: np.ndarray,
    gate_name: Literal["CX"],
    index: tuple[int, int],
) -> np.ndarray:
    # Example:
    #   - n = 4
    #   - index = (1, 3)
    #
    # The subscripts of `np.einsum()` are
    #   - `state`: [0, 1, 2, 3]
    #   - `gate`: [4, 5, 1, 3]
    #   - output: [0, 4, 2, 5]

    gate = _two_qubit_gates[gate_name]
    n = state.ndim

    return np.einsum(
        state,
        [x for x in range(n)],
        gate,
        [n, n + 1, index[0], index[1]],
        [n if x == index[0] else n + 1 if x == index[1] else x for x in range(n)],
    )


def apply_gate(
    state: np.ndarray,
    gate_name: Literal["I", "X", "H", "CX"],
    index: int | tuple[int, int],
) -> np.ndarray:
    if gate_name in _one_qubit_gates:
        return apply_single_qubit_gate(state, gate_name, index)
    return apply_two_qubit_gate(state, gate_name, index)
