import numpy as np

from tensor import (
    H,
    I,
    kron,
    X,
    apply_single_qubit_gate,
    apply_two_qubit_gate,
)

zero = np.array([1.0, 0.0])  # |0>
one = np.array([0.0, 1.0])  # |1>
plus = 1 / np.sqrt(2) * (zero + one)  # |+>
minus = 1 / np.sqrt(2) * (zero - one)  # |->


def test_I():
    np.testing.assert_array_equal(I @ zero, zero)  # I |0> = |0>
    np.testing.assert_array_equal(I @ one, one)  # I |1> = |1>


def test_X():
    np.testing.assert_array_equal(X @ zero, one)  # X |0> = |1>
    np.testing.assert_array_equal(X @ one, zero)  # X |1> = |0>


def test_H():
    np.testing.assert_array_equal(H @ zero, plus)  # H |0> = |+>
    np.testing.assert_array_equal(H @ one, minus)  # H |1> = |->
    np.testing.assert_array_almost_equal(H @ H @ zero, zero)  # H H |0> = |0>
    np.testing.assert_array_almost_equal(H @ H @ one, one)  # H H |1> = |1>


def test_CX():
    # Tested in apply_two_qubit_gate.
    pass


def test_kron():
    np.testing.assert_array_equal(kron(zero), zero)
    np.testing.assert_array_equal(
        kron(zero, zero),
        np.array(
            [
                [1.0, 0.0],
                [0.0, 0.0],
            ]
        ),
    )
    np.testing.assert_array_equal(
        kron(one, zero),
        np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
            ]
        ),
    )
    np.testing.assert_array_equal(
        kron(one, zero, one),
        np.array(
            [
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
                [
                    [0.0, 1.0],
                    [0.0, 0.0],
                ],
            ]
        ),
    )


def test_apply_single_qubit_gate():
    np.testing.assert_array_equal(
        apply_single_qubit_gate(kron(zero, zero, zero), "X", 1), kron(zero, one, zero)
    )
    np.testing.assert_array_equal(
        apply_single_qubit_gate(kron(zero, zero, zero), "X", 2), kron(zero, zero, one)
    )
    np.testing.assert_array_equal(
        apply_single_qubit_gate(kron(zero, zero, zero), "H", 0), kron(plus, zero, zero)
    )


def test_apply_two_qubit_gate():
    np.testing.assert_array_equal(
        apply_two_qubit_gate(kron(zero, zero, zero), "CX", (1, 2)),
        kron(zero, zero, zero),
    )
    np.testing.assert_array_equal(
        apply_two_qubit_gate(kron(one, zero, zero), "CX", (0, 2)), kron(one, zero, one)
    )
