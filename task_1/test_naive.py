import numpy as np

from naive import (
    CX,
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
    # CX |00> = |00>
    np.testing.assert_array_equal(CX @ kron(zero, zero), kron(zero, zero))
    # CX |01> = |01>
    np.testing.assert_array_equal(CX @ kron(zero, one), kron(zero, one))
    # CX |10> = |11>
    np.testing.assert_array_equal(CX @ kron(one, zero), kron(one, one))
    # CX |11> = |10>
    np.testing.assert_array_equal(CX @ kron(one, one), kron(one, zero))


def test_kron():
    np.testing.assert_array_equal(kron(zero), zero)
    np.testing.assert_array_equal(kron(zero, zero), np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(kron(one, zero), np.array([0.0, 0.0, 1.0, 0.0]))


def test_apply_single_qubit_gate():
    np.testing.assert_array_equal(apply_single_qubit_gate(zero, "H", 0), plus)
    np.testing.assert_array_equal(
        apply_single_qubit_gate(kron(zero, zero), "H", 1), kron(zero, plus)
    )


def test_apply_two_qubit_gate():
    np.testing.assert_array_equal(
        apply_two_qubit_gate(kron(one, zero), "CX", (0, 1)), kron(one, one)
    )
    np.testing.assert_array_equal(
        apply_two_qubit_gate(kron(zero, one, zero, zero), "CX", (1, 2)),
        kron(zero, one, one, zero),
    )
