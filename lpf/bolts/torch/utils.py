def same_padding(i: int, k: int, stride: int = 1) -> int:
    assert k % 2 == 1
    p: int = (stride * (i - 1) - i + k) // 2
    return p

 