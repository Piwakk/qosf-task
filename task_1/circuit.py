from random import choice


def get_random_gate(n: int) -> tuple[str, int | tuple[int, int]]:
    """Return a `(gate_name, index)`."""
    available_gates = ["X", "H"]
    if n >= 2:
        available_gates.append("CX")

    gate_name = choice(available_gates)

    if gate_name == "CX":
        index = choice(range(n - 1))
        return gate_name, (index, index + 1)

    index = choice(range(n))
    return gate_name, index


def get_random_circuit(n: int, depth: int) -> list[tuple[str, int | tuple[int, int]]]:
    """Return a list of `(gate_name, index)`."""
    return [get_random_gate(n) for _ in range(depth)]
