-- Q1 selection: raw readings of 10 sensors for one station over one day.
SELECT "time", id_station, s32, s15, s63, s97, s57, s60, s83, s48, s26, s12 FROM d1
 WHERE id_station IN ('st2')
   AND "time" > TIMESTAMP '2019-03-01 13:15:18' - INTERVAL '1 day'
   AND "time" < TIMESTAMP '2019-03-01 13:15:18';
-- Q2 filtering: Q1 restricted to readings where the first sensor is above 0.95.
SELECT "time", id_station, s32, s15, s63, s97, s57, s60, s83, s48, s26, s12 FROM d1
 WHERE id_station IN ('st2')
   AND "time" > TIMESTAMP '2019-03-01 13:15:18' - INTERVAL '1 day'
   AND "time" < TIMESTAMP '2019-03-01 13:15:18'
   AND (s32 > 0.95);
-- Q3 aggregation: per-station average of 10 sensors over one day.
SELECT id_station, avg(s32), avg(s15), avg(s63), avg(s97), avg(s57), avg(s60), avg(s83), avg(s48), avg(s26), avg(s12) FROM d1
 WHERE "time" > TIMESTAMP '2019-03-01 13:15:18' - INTERVAL '1 day'
   AND "time" < TIMESTAMP '2019-03-01 13:15:18'
   AND id_station IN ('st2')
 GROUP BY id_station;
-- Q4 downsampling: hourly averages of 10 sensors over one day.
SELECT id_station, EXTRACT(YEAR FROM "time") AS "year", date_trunc('month', "time") AS "month",
       date_trunc('day', "time") AS "day", date_trunc('hour', "time") AS "hour", avg(s32), avg(s15), avg(s63), avg(s97), avg(s57), avg(s60), avg(s83), avg(s48), avg(s26), avg(s12) FROM d1
 WHERE "time" > TIMESTAMP '2019-03-01 13:15:18' - INTERVAL '1 day'
   AND "time" < TIMESTAMP '2019-03-01 13:15:18'
   AND id_station IN ('st2')
 GROUP BY id_station, "year", "month", "day", "hour";
-- Q5 upsampling: 10-second readings filled to a 5-second grid by linear interpolation.
-- TSM-Bench uses engine-specific gap filling here. Because d1 is sampled every 10 seconds,
-- each new point is the midpoint of two neighbouring readings, which LEAD expresses portably.
WITH w AS (
  SELECT "time", id_station, s32, s15, s63, s97, s57, s60, s83, s48, s26, s12,
         LEAD(s32) OVER (PARTITION BY id_station ORDER BY "time") AS next_s32,
         LEAD(s15) OVER (PARTITION BY id_station ORDER BY "time") AS next_s15,
         LEAD(s63) OVER (PARTITION BY id_station ORDER BY "time") AS next_s63,
         LEAD(s97) OVER (PARTITION BY id_station ORDER BY "time") AS next_s97,
         LEAD(s57) OVER (PARTITION BY id_station ORDER BY "time") AS next_s57,
         LEAD(s60) OVER (PARTITION BY id_station ORDER BY "time") AS next_s60,
         LEAD(s83) OVER (PARTITION BY id_station ORDER BY "time") AS next_s83,
         LEAD(s48) OVER (PARTITION BY id_station ORDER BY "time") AS next_s48,
         LEAD(s26) OVER (PARTITION BY id_station ORDER BY "time") AS next_s26,
         LEAD(s12) OVER (PARTITION BY id_station ORDER BY "time") AS next_s12
    FROM d1
   WHERE id_station IN ('st2')
     AND "time" > TIMESTAMP '2019-03-01 13:15:18' - INTERVAL '1 day'
     AND "time" < TIMESTAMP '2019-03-01 13:15:18'
)
SELECT "time", id_station, s32, s15, s63, s97, s57, s60, s83, s48, s26, s12 FROM w
UNION ALL
SELECT "time" + INTERVAL '5 seconds' AS "time", id_station,
       (s32 + next_s32) / 2 AS s32,
       (s15 + next_s15) / 2 AS s15,
       (s63 + next_s63) / 2 AS s63,
       (s97 + next_s97) / 2 AS s97,
       (s57 + next_s57) / 2 AS s57,
       (s60 + next_s60) / 2 AS s60,
       (s83 + next_s83) / 2 AS s83,
       (s48 + next_s48) / 2 AS s48,
       (s26 + next_s26) / 2 AS s26,
       (s12 + next_s12) / 2 AS s12
  FROM w
 WHERE next_s32 IS NOT NULL
 ORDER BY "time";
-- Q6 average: row-wise average of two sensors for one station over one day.
WITH data1 AS (
  SELECT "time", s32 AS s_1, s15 AS s_2 FROM d1
   WHERE "time" > TIMESTAMP '2019-03-01 13:15:18' - INTERVAL '1 day'
     AND "time" < TIMESTAMP '2019-03-01 13:15:18'
     AND id_station IN ('st2')
)
SELECT data1."time", s_1, s_2, (s_1 + s_2) / 2 AS avg FROM data1;
-- Q7 correlation: correlation of two sensors for one station over one day.
SELECT corr(s32, s15) FROM d1
 WHERE id_station IN ('st2')
   AND "time" > TIMESTAMP '2019-03-01 13:15:18' - INTERVAL '1 day'
   AND "time" < TIMESTAMP '2019-03-01 13:15:18';
