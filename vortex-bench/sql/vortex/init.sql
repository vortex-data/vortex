-- Script that prepares Parquet data for our SQL microbenchmarks.

COPY (
    SELECT
        i % 1000 AS col,
        (i * 2654435761) % 100000 AS col2
    FROM range(250000000) t(i)
) TO 'test.parquet' (FORMAT parquet);
