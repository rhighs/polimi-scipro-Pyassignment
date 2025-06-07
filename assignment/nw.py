import os
import argparse
import numpy as np

from concurrent.futures import ProcessPoolExecutor

GAP_PENALTY_DEFAULT = -4
MATCH_DEFAULT = 1
MISMATCH_DEFAULT = -1


def custom_score(a, b, match, mismatch):
    return match if a == b else mismatch


def _compute_cell(args):
    i, j, seq1, seq2, dp_snapshot, match, mismatch, gap_penalty = args
    match_score = dp_snapshot[i - 1, j - 1] + custom_score(
        seq1[j - 1], seq2[i - 1], match, mismatch
    )
    delete = dp_snapshot[i - 1, j] + gap_penalty
    insert = dp_snapshot[i, j - 1] + gap_penalty
    return (i, j, max(match_score, delete, insert))


def nw_parallel(
    seq1: str,
    seq2: str,
    nproc: int = os.cpu_count() or 1,
    match: int = MATCH_DEFAULT,
    mismatch: int = MISMATCH_DEFAULT,
    gap_penalty: int = GAP_PENALTY_DEFAULT,
):
    m, n = len(seq2), len(seq1)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    dp[:, 0] = np.arange(m + 1) * gap_penalty
    dp[0, :] = np.arange(n + 1) * gap_penalty

    if nproc is None or nproc < 1:
        nproc = os.cpu_count() or 1

    for k in range(2, m + n + 1):
        cells = [(i, k - i) for i in range(1, min(k, m + 1)) if 1 <= (k - i) <= n]
        dp_snapshot = dp.copy()
        with ProcessPoolExecutor(max_workers=nproc) as executor:
            results = list(
                executor.map(
                    _compute_cell,
                    [
                        (i, j, seq1, seq2, dp_snapshot, match, mismatch, gap_penalty)
                        for i, j in cells
                    ],
                )
            )
        for i, j, value in results:
            dp[i, j] = value

    aligned_seq1, aligned_seq2 = [], []
    i, j = m, n

    while i > 0 and j > 0:
        _score = dp[i, j]
        if _score == dp[i - 1, j - 1] + custom_score(
            seq1[j - 1], seq2[i - 1], match, mismatch
        ):
            aligned_seq1.append(seq1[j - 1])
            aligned_seq2.append(seq2[i - 1])
            i -= 1
            j -= 1
        elif _score == dp[i - 1, j] + gap_penalty:
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
        description="Parallel Needleman-Wunsch global sequence aligner with configurable scoring"
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
    parser.add_argument(
        "--match", type=int, default=MATCH_DEFAULT, help="Match score (default: 1)"
    )
    parser.add_argument(
        "--mismatch",
        type=int,
        default=MISMATCH_DEFAULT,
        help="Mismatch score (default: -1)",
    )
    parser.add_argument(
        "--gap-penalty",
        type=int,
        default=GAP_PENALTY_DEFAULT,
        help="Gap penalty (default: -4)",
    )
    args = parser.parse_args()

    a1, a2, s = nw_parallel(
        args.seq1, args.seq2, args.nproc, args.match, args.mismatch, args.gap_penalty
    )
    print(a1)
    print(a2)
    print(f"alignment score: {s}")


if __name__ == "__main__":
    main()
