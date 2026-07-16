-- As this aggregation has a filter, Vortex has to use a linear scan. Once stats
-- are propagated to arrays, this should use zone maps to aggregate instead of
-- decoding and processing each row.
SELECT sum(col) FROM test WHERE col2 > 0 AND col2 < 1000;
