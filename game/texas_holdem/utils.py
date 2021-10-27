from functools import reduce


def n_combinations(a: int, b: int) -> int:
    """# of combinations
    C^b_a
    """
    numerator = reduce(lambda x, y: x * y, range(a, a - b, -1), 1)
    denominator = reduce(lambda x, y: x * y, range(b, 0, -1), 1)
    return numerator // denominator
