import numpy as np

############## functions #############

def circular_shift_identity(N, e):
    """
    Returns the circular right-shift of the identity matrix I_N by e positions.

    Parameters:
    - N (int): The size of the identity matrix. Must be an integer ≥ 2.
    - e (int): The number of positions to circularly right-shift (0 ≤ e ≤ N - 1).

    Returns:
    - numpy.ndarray: The circulant permutation matrix (CPM), obtained by circularly right-shifting every row of the identity matrix of size N x N by e positions.
    """

    if not isinstance(N, int) or N < 2:
        raise ValueError("N must be an integer ≥ 2.")
    if not isinstance(e, int) or e < 0 or e > N - 1:
        raise ValueError("e must be an integer such that 0 ≤ e ≤ N - 1.")

    I = np.eye(N, dtype=int)

    # Circularly shift every row of the identity matrix to the right by e positions
    CPM = np.roll(I, shift=e, axis=1)

    return CPM

import numpy as np

def circular_shift_identity(N, e):
    """
    Returns the circular right-shift of the identity matrix I_N by e positions.

    Parameters:
    - N (int): Size of the identity matrix (N ≥ 1)
    - e (int): Number of positions to shift (0 ≤ e ≤ N - 1)

    Returns:
    - numpy.ndarray: N x N matrix with each row shifted right by e positions
    """
    if not isinstance(N, int) or N < 1:
        raise ValueError("N must be an integer ≥ 1.")
    if not isinstance(e, int) or not (0 <= e <= N - 1):
        raise ValueError(f"e must be an integer between 0 and {N - 1}.")

    I = np.eye(N, dtype=int)
    return np.roll(I, shift=e, axis=1)

########

def PCM_QCLDPC(E, N):
    """
    Constructs a parity check matrix H from an exponent matrix E and lifting factor N.

    Rules:
    - Entry -1 in E → N x N all-zero matrix
    - Entry e in E taking values in {0, ..., N-1} → circularly right-shifted identity matrix by e positions

    Parameters:
    - E (2D list or numpy.ndarray): Exponent matrix with values in {-1, 0, ..., M-1}, for some M
    - N (int): Lifting factor (N ≥ 1)

    Returns:
    - numpy.ndarray: Parity-check matrix H of shape (N * rows(E), N * cols(E))
    """

    if not isinstance(N, int) or N < 1:
        raise ValueError("N must be an integer ≥ 1.")

    E = np.array(E)
    M = E.max() + 1

    if not np.issubdtype(E.dtype, np.integer) or not np.all(E >= -1):
        raise ValueError("All entries in E must be integers ≥ -1.")


    H_rows = []
    for row in E:
        block_row = []
        for e in row:
            if e == -1:
                block = np.zeros((N, N), dtype=int)
            else:
                block = circular_shift_identity(N, e)
            block_row.append(block)
        H_rows.append(np.hstack(block_row))  # Concatenate horizontally

    H = np.vstack(H_rows)  # Concatenate rows vertically
    
    return H



##################### C2 ############################

# QC-LDPC code C2(3224, 1612), (4, 8)-regular

n2 = 3224
k2 = 1612
N = 403 # listing factor

# Exponent matrix for the QC-LDPC code C2(3224, 1612) 
E_C2 = np.array([
    [345, 152,  72, 376, 377, 197,   4, 144],
    [187, 398, 320, 225, 330, 198,  79, 289],
    [271, 165, 259, 105, 288, 254,  51, 236],
    [111, 233, 380, 332,  47,  76, 222, 247],
])

H2 = PCM_QCLDPC(E_C2, N2)

##################### C3 ############################

# QC-LDPC code C3(4016, 2761), (5, 16)-regular
n3 = 4016
k3 = 2761
N = 251  # lifting factor

# Left part of exponent matrix 

E_C3_left = np.array([
    [  6,  98, 208, 177,  76,  76,  76,  48],
    [198,  42, 155, 127,  29,  32,  35,  10],
    [ 31, 211, 158,   0, 238, 111, 199,   8],
    [117,  51,   3,  65,  57, 150, 243,  57],
    [181, 142, 121, 210, 229,  98, 218,  59],
])

# Right part of exponent matrix 
E_C3_right = np.array([
    [111,  76,  76,  34,  76,  76,  64,  85],
    [ 76,  44,  47,   8,  53,  56,  47,  71],
    [195, 248, 121, 167,  46, 170, 246, 140],
    [213,  20, 113, 164,  48, 141, 222,  85],
    [242,  76, 196,  23, 185,  54, 162,  52],
])

# Full exponent matrix for C3
E_C3 = np.hstack((E_C3_left, E_C3_right))
H3 = PCM_QCLDPC(E_C3, N3)
