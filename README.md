# A parallel Needleman-Wunsch implementation using antidiagonals

## Usage

### 1. Install requirements

```bash
pip install poetry
poetry install
```

### 2. Command-line usage

```bash
poetry run nw <seq1> <seq2> <optional-nproc> [--match MATCH] [--mismatch MISMATCH] [--gap-penalty GAP_PENALTY]
```

#### **Parameters**

| Argument           | Type   | Description                                                                                              | Default  |
|--------------------|--------|----------------------------------------------------------------------------------------------------------|----------|
| `<seq1>`           | str    | First sequence to align                                                                                   | Required |
| `<seq2>`           | str    | Second sequence to align                                                                                  | Required |
| `<optional-nproc>` | int    | Number of processes to use for parallelization. Defaults to the number of CPU cores.                      | Optional |
| `--match`          | int    | Score for a match between characters.                                                                     | 1        |
| `--mismatch`       | int    | Score for a mismatch between characters.                                                                  | -1       |
| `--gap-penalty`    | int    | Penalty for a gap (insertion/deletion).                                                                   | -4       |

#### Example
```bash
$ poetry run nw ACGTGGTA AGTTGTA 4 --match 2 --mismatch -3 --gap-penalty -5
ACGTGGTA
A-GTTGTA
alignment score: 4
```

---

## Parallelization strategy

The core bottleneck in Needleman-Wunsch is filling the DP matrix, where each cell depends on its top, left, and top-left neighbors. Direct parallelization is impossible due to these dependencies. **Cells along the same antidiagonal (where `i + j` is constant)** only depend on cells from earlier antidiagonals and can be computed in parallel.

- The DP matrix is processed antidiagonal by antidiagonal.
- For each antidiagonal, all cells are assigned to worker processes and computed simultaneously.
- Each cell `(i, j)` is computed using:
    1. `dp[i-1, j-1] + score(seq1[j-1], seq2[i-1])` (match/mismatch)
    2. `dp[i-1, j] + GAP_PENALTY` (deletion)
    3. `dp[i, j-1] + GAP_PENALTY` (insertion)
- Once an antidiagonal is complete, the next can be processed.