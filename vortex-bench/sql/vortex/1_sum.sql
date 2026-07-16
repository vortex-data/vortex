-- When Footer changes land, vortex-duckdb should populate statistics from
-- Footer without loading and decoding the data.
SELECT sum(col) FROM test;
