"""Convolve a user-entered 3x3 matrix with a user-entered 2x2 kernel."""

import numpy as np
from scipy.signal import correlate2d as scipy_correlate2d


def convolve2d(mat, ker):
    """Convolve mat with ker (valid mode, no padding).
    Uses cross-correlation (no kernel flip) — matches scipy.signal.correlate2d.
    """
    m_rows, m_cols = mat.shape
    k_rows, k_cols = ker.shape

    out_rows = m_rows - k_rows + 1
    out_cols = m_cols - k_cols + 1

    output = np.zeros((out_rows, out_cols))

    for i in range(out_rows):
        for j in range(out_cols):
            region = mat[i:i + k_rows, j:j + k_cols]
            output[i, j] = np.sum(region * ker)

    return output


def read_matrix(rows, cols, label):
    """Prompt the user to enter a matrix row by row."""
    print(f"\nEnter {label} ({rows}x{cols}), one row at a time, values separated by spaces:")
    data = []
    for i in range(rows):
        while True:
            raw = input(f"  Row {i + 1}: ").strip()
            values = raw.split()
            if len(values) == cols:
                try:
                    data.append([float(v) for v in values])
                    break
                except ValueError:
                    print(f"  Please enter {cols} numbers.")
            else:
                print(f"  Expected {cols} values, got {len(values)}. Try again.")
    return np.array(data)


matrix = read_matrix(3, 3, "3x3 matrix")
kernel = read_matrix(2, 2, "2x2 kernel")

result = convolve2d(matrix, kernel)

print("\nMatrix (3x3):")
print(matrix)
print("\nKernel (2x2):")
print(kernel)
print("\nConvolution result (2x2):")
print(result)

scipy_result = scipy_correlate2d(matrix, kernel, mode='valid')
print("\nScipy verification (2x2):")
print(scipy_result)
print("\nResults match:", np.allclose(result, scipy_result))
