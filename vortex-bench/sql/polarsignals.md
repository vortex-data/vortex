# PolarSignals profiling benchmark

A synthetic continuous-profiling workload with ten nullable dictionary-string
labels. Fill rates range from 0% to 100%, with cardinalities from 4 to 800.
Consecutive samples repeat label sets, with lowercase values such as `comm_0`.
NULL labels do not match positive or negative string predicates.

Stacks are lists of location structs containing lists of line structs, with
dictionary-encoded scalar children. Profile metadata uses dictionary strings,
period and duration use Int64, and timestamps use nanoseconds in UTC.

The queries exercise:

- Narrow and wide time windows with profile metadata filters and stack projection.
- Raw equality and regular expressions on a label.
- Time-bucket sums, label grouping, distinct labels, mapping files and profile types.
- A positive conjunction of lowercase equality, substring, prefix, suffix and
  membership filters across labels, with stack projection.
- A negative conjunction of lowercase inequality, absent substring, prefix,
  suffix and nonmembership filters, with time-bucket sums.
- An equivalent positive ILIKE conjunction with the same stack projection.
  Escaped underscores match literal underscores rather than wildcard characters.

At scale factor 1, the generator writes one million rows in deterministic
10,000-row batches. Parquet uses Zstd level 3. Generic conversion writes standard
or compact Vortex files, whose physical encodings are selected by the writer.

Parquet-versus-Vortex timings include representation differences. String filters
can execute above the scan, so inspect physical plans separately from timings.
The string values are lowercase ASCII. Lowercase filters exercise function
evaluation but do not change these values or measure Unicode performance.

## CI variant

CI runs this suite from local NVMe as the `PolarSignals Profiling` PR comment.
It runs DataFusion over Vortex only, without a Parquet drift control.

## Running locally

```bash
vx-bench run polarsignals --engine datafusion --format vortex
```
