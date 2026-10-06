k = [99, 11, 12, 13]
v = [20, 21, 22, 23]
print("K:", k)
print("V:", v)
k_cache = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
]
slot = 2
k_cache[slot] = k.copy()

print("K cache:", k_cache)