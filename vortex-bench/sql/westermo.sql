-- SPDX-License-Identifier: Apache-2.0
-- SPDX-FileCopyrightText: Copyright the Vortex contributors

-- `ts` is int64 milliseconds since the start of collection. Windows used below:
-- day 10 = [864000000, 950400000), a 6 hour window = [864000000, 885600000),
-- a 1 hour window = [871200000, 874800000), week 2 = [604800000, 1209600000).
-- A step of N ms buckets a sample with `ts - ts % N`. Every query runs unchanged on DataFusion
-- and DuckDB, and regexes are anchored because DuckDB's `~` matches the whole string.
-- Each query is a PromQL expression translated to SQL, given in the comment above it.

-- Q0: cpu_user{instance="system-7"}[1h], a raw range read of one series.
SELECT ts, value
  FROM samples
 WHERE labels.__name__ = 'cpu_user'
   AND labels.instance = 'system-7'
   AND ts >= 871200000 AND ts < 874800000
 ORDER BY ts;

-- Q1: avg by (instance) (load_15m) at a 5m step over a day.
SELECT ts - ts % 300000 AS bucket, labels.instance AS instance, avg(value) AS value
  FROM samples
 WHERE labels.__name__ = 'load_15m'
   AND ts >= 864000000 AND ts < 950400000
 GROUP BY bucket, instance
 ORDER BY bucket, instance;

-- Q2: max by (instance) (cpu_user{instance=~"system-1[0-9]"}) at a 1m step over 6 hours.
SELECT ts - ts % 60000 AS bucket, labels.instance AS instance, max(value) AS value
  FROM samples
 WHERE labels.__name__ = 'cpu_user'
   AND labels.instance ~ '^system-1[0-9]$'
   AND ts >= 864000000 AND ts < 885600000
 GROUP BY bucket, instance
 ORDER BY bucket, instance;

-- Q3: sum by (__name__) ({__name__=~"disk_.*", instance!~"system-(1|2)"}) at a 1h step over a week.
SELECT ts - ts % 3600000 AS bucket, labels.__name__ AS metric, sum(value) AS value
  FROM samples
 WHERE labels.__name__ ~ '^disk_.*$'
   AND labels.instance !~ '^system-(1|2)$'
   AND ts >= 604800000 AND ts < 1209600000
 GROUP BY bucket, metric
 ORDER BY bucket, metric;

-- Q4: avg by (__name__) ({__name__=~"cpu_.*"}) at a 15m step over a day.
SELECT ts - ts % 900000 AS bucket, labels.__name__ AS metric, avg(value) AS value
  FROM samples
 WHERE labels.__name__ ~ '^cpu_.*$'
   AND ts >= 864000000 AND ts < 950400000
 GROUP BY bucket, metric
 ORDER BY bucket, metric;

-- Q5: min_over_time(sys_mem_available[30d]) over the whole data set.
SELECT labels.instance AS instance, min(value) AS value
  FROM samples
 WHERE labels.__name__ = 'sys_mem_available'
 GROUP BY instance
 ORDER BY instance;

-- Q6: 1 - sys_mem_available / sys_mem_total, one-to-one vector matching on instance and time,
-- averaged per hour over a day.
WITH avail AS (
  SELECT labels.instance AS instance, ts, value
    FROM samples
   WHERE labels.__name__ = 'sys_mem_available'
     AND ts >= 864000000 AND ts < 950400000
), total AS (
  SELECT labels.instance AS instance, ts, value
    FROM samples
   WHERE labels.__name__ = 'sys_mem_total'
     AND ts >= 864000000 AND ts < 950400000
)
SELECT avail.ts - avail.ts % 3600000 AS bucket, avail.instance AS instance,
       avg(1 - avail.value / total.value) AS used_ratio
  FROM avail
  JOIN total ON avail.instance = total.instance AND avail.ts = total.ts
 GROUP BY bucket, avail.instance
 ORDER BY bucket, instance;

-- Q7: count_over_time((load_1m > 2)[30d]) over the whole data set.
SELECT labels.instance AS instance, count(*) AS samples
  FROM samples
 WHERE labels.__name__ = 'load_1m'
   AND value > 2.0
 GROUP BY instance
 ORDER BY instance;

-- Q8: the last sample of every series, the newest point per series.
WITH series AS (
  SELECT labels.__name__ AS metric, labels.instance AS instance, ts, value
    FROM samples
), newest AS (
  SELECT metric, instance, max(ts) AS last_ts
    FROM series
   GROUP BY metric, instance
)
SELECT series.metric AS metric, series.instance AS instance, series.ts AS last_ts, series.value AS value
  FROM series
  JOIN newest
    ON series.metric = newest.metric
   AND series.instance = newest.instance
   AND series.ts = newest.last_ts
 ORDER BY metric, instance;

-- Q9: label_values(instance).
SELECT DISTINCT labels.instance AS instance
  FROM samples
 ORDER BY instance;

-- Q10: label_values(__name__), the metric names API.
SELECT DISTINCT labels.__name__ AS metric
  FROM samples
 ORDER BY metric;

-- Q11: label_values(sys_thermal, instance), label values scoped by a matcher.
SELECT DISTINCT labels.instance AS instance
  FROM samples
 WHERE labels.__name__ = 'sys_thermal'
 ORDER BY instance;

-- Q12: series API with match[]={__name__=~"sys_mem_.*"} over a day.
SELECT DISTINCT labels
  FROM samples
 WHERE labels.__name__ ~ '^sys_mem_.*$'
   AND ts >= 864000000 AND ts < 950400000;

-- Q13: max(cpu_user) at a 1m step, the newest 5 buckets before day 20.
SELECT ts - ts % 60000 AS bucket, max(value) AS value
  FROM samples
 WHERE labels.__name__ = 'cpu_user'
   AND ts < 1728000000
 GROUP BY bucket
 ORDER BY bucket DESC
 LIMIT 5;

-- Q14: count_over_time(server_up[1d]) with first and last sample, which surfaces scrape gaps.
SELECT labels.instance AS instance, count(*) AS samples, min(ts) AS first_ts, max(ts) AS last_ts
  FROM samples
 WHERE labels.__name__ = 'server_up'
   AND ts >= 864000000 AND ts < 950400000
 GROUP BY instance
 ORDER BY instance;
