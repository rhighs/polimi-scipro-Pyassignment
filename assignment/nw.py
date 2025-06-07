import os
import argparse
import numpy as np

from typing import cast
from Bio.Align import substitution_matrices
from concurrent.futures import ProcessPoolExecutor

pam = substitution_matrices.load("PAM70")
GAP_PENALTY = -4


def score(a, b) -> int:
    return cast(int, pam.get((a.upper(), b.upper())))


def _compute_cell(args):
    i, j, seq1, seq2, dp_snapshot = args
    match = dp_snapshot[i - 1, j - 1] + score(seq1[j - 1], seq2[i - 1])
    delete = dp_snapshot[i - 1, j] + GAP_PENALTY
    insert = dp_snapshot[i, j - 1] + GAP_PENALTY
    return (i, j, max(match, delete, insert))


def nw_parallel(seq1: str, seq2: str, nproc: int = os.cpu_count() or 1):
    m, n = len(seq2), len(seq1)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    dp[:, 0] = np.arange(m + 1) * GAP_PENALTY
    dp[0, :] = np.arange(n + 1) * GAP_PENALTY

    for k in range(2, m + n + 1):
        cells = [(i, k - i) for i in range(1, min(k, m + 1)) if 1 <= (k - i) <= n]
        # take a snapshot of the current dp state for this antidiagonal
        dp_snapshot = dp.copy()
        with ProcessPoolExecutor(max_workers=nproc) as executor:
            results = list(
                executor.map(
                    _compute_cell, [(i, j, seq1, seq2, dp_snapshot) for i, j in cells]
                )
            )
        for i, j, value in results:
            dp[i, j] = value

    aligned_seq1, aligned_seq2 = [], []
    i, j = m, n

    while i > 0 and j > 0:
        _score = dp[i, j]
        if _score == dp[i - 1, j - 1] + score(seq1[j - 1], seq2[i - 1]):
            aligned_seq1.append(seq1[j - 1])
            aligned_seq2.append(seq2[i - 1])
            i -= 1
            j -= 1
        elif _score == dp[i - 1, j] + GAP_PENALTY:
            aligned_seq1.append("-")
            aligned_seq2.append(seq2[i - 1])
            i -= 1
        else:
            aligned_seq1.append(seq1[j - 1])
            aligned_seq2.append("-")
            j -= 1

    while j > 0:
        aligned_seq1.append(seq1[j - 1])
        aligned_seq2.append("-")
        j -= 1
    while i > 0:
        aligned_seq1.append("-")
        aligned_seq2.append(seq2[i - 1])
        i -= 1

    aligned_seq1 = "".join(reversed(aligned_seq1))
    aligned_seq2 = "".join(reversed(aligned_seq2))
    return aligned_seq1, aligned_seq2, dp[m, n]


def main():
    parser = argparse.ArgumentParser(
        description="Parallel Needleman-Wunsch global sequence aligner using PAM70"
    )
    parser.add_argument("seq1", type=str, help="First sequence to align")
    parser.add_argument("seq2", type=str, help="Second sequence to align")
    parser.add_argument(
        "nproc",
        type=int,
        nargs="?",
        default=None,
        help="Number of parallel workers (default: number of CPU cores)",
    )
    args = parser.parse_args()

    a1, a2, s = nw_parallel(args.seq1, args.seq2, args.nproc)
    print(a1)
    print(a2)
    print(f"alignment score: {s}")


if __name__ == "__main__":
    main()
