-- Q0: Full scan, sum all float columns (decompression throughput)
SELECT SUM(sensor_reading), SUM(price), SUM(measurement), SUM(temperature), SUM(velocity) FROM alp_floats;
-- Q1: Filtered scan on id range (predicate pushdown + partial decompress)
SELECT SUM(sensor_reading), AVG(price) FROM alp_floats WHERE id BETWEEN 100000 AND 200000;
-- Q2: Group-by aggregation over a low-cardinality key
SELECT label, AVG(sensor_reading), AVG(price), AVG(temperature) FROM alp_floats GROUP BY label;
-- Q3: Filter on a float column value range
SELECT COUNT(*), AVG(velocity) FROM alp_floats WHERE temperature > 22.0 AND temperature < 23.0;
-- Q4: Multi-column projection with filter
SELECT id, sensor_reading, price FROM alp_floats WHERE velocity > 299.0 AND velocity < 301.0;
-- Q5: ORDER BY on a float column with LIMIT (top-k)
SELECT id, measurement FROM alp_floats ORDER BY measurement DESC LIMIT 100;
-- Q6: Heavy aggregation — group by label, multiple agg functions
SELECT label, MIN(sensor_reading), MAX(sensor_reading), AVG(price), SUM(measurement), COUNT(*) FROM alp_floats GROUP BY label ORDER BY label;
-- Q7: Scan with arithmetic expression on compressed columns
SELECT SUM(sensor_reading * price + measurement) FROM alp_floats;
