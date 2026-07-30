-- Script that prepares Parquet data for our SQL microbenchmarks.

COPY (
    SELECT
        i AS id,
        (i % 1000)::INTEGER AS col,
        ((i * 2654435761) % 100000)::INTEGER AS col2
    FROM range(2500000000) t(i)
) TO 'test.parquet' (FORMAT parquet);
