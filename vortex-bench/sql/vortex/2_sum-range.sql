-- Some zone maps are pruned, two zones are included partially, and all others
-- are included. Only two zone maps should be decoded fully.
SELECT sum(col) FROM test WHERE id >= 400000123 AND id < 2100000456;
