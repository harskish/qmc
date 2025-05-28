## QMC visualizer

### Sequences:
| Name                  | Random access (uint32 idx) | Fixed sample count |
| --------              | -------                    | -------            |
| Murmurhash (prng)     | O(1)                       | -                  |
| Sobol                 | O(1)                       | -                  |
| Sobol + CP            | O(1)                       | -                  |
| Sobol + RDS           | O(1)                       | -                  |
| Sobol + Owen-scramble | O(1)                       | -                  |
| Cascaded Sobol        | ?                          | ~ (perf)           |
| Hammersley            | O(log_b(N))                | x                  |
| Halton                | O(log_b(N))                | -                  |
| Leaped Halton         | O(log_b(N))                | -                  |
| R2                    | O(1)                       | -                  |

- CP = Cranley-Patterson rotation: `(v + rand()) % 1`
- RDS = Random digit scrambling: `v xor rand()`

### About complexity
For practical use, the range of the index/dimension can be assumed to be uint32. Under such an assumption, any complexities have a predictable constant worst-case! If using BigInt or similar, then Sobol sampling (logN), owen scrambling (loglogN?) are no longer constant-time.