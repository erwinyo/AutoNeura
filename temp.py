

initial_perm = [58, 50, 42, 34, 26, 18, 10, 2,
				60, 52, 44, 36, 28, 20, 12, 4,
				62, 54, 46, 38, 30, 22, 14, 6,
				64, 56, 48, 40, 32, 24, 16, 8,
				57, 49, 41, 33, 25, 17, 9, 1,
				59, 51, 43, 35, 27, 19, 11, 3,
				61, 53, 45, 37, 29, 21, 13, 5,
				63, 55, 47, 39, 31, 23, 15, 7]


def lookup_mapping_permutation(k, arr, sz):
    permutation = ""
    for i in range(0, sz):
        permutation = permutation + k[arr[i] - 1]
    return permutation



def main():
    permutated = lookup_mapping_permutation(
        "0001001000110100010101101010101111001101000100110010010100110110",
        initial_perm,
        64
    )

    print("Permutated: ", permutated)

if __name__ == "__main__":
    main()