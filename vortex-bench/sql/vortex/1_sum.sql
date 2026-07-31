-- All zone maps are included since we don't have a filter. Vortex doesn't load
-- data but instead populates Sum() stat from zone maps
SELECT sum(col) FROM test;
