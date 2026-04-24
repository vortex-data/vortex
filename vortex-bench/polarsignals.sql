-- Q0: Narrow time range with filters.
SELECT locations, value, period, time_nanos, labels.comm FROM stacktraces
 WHERE time_nanos >= arrow_cast(0, 'Timestamp(Nanosecond, Some("UTC"))')
   AND time_nanos <= arrow_cast(50000000000, 'Timestamp(Nanosecond, Some("UTC"))')
   AND producer = 'parca_agent'
   AND sample_type = 'samples'
   AND sample_unit = 'count'
   AND period_type = 'cpu'
   AND period_unit = 'nanoseconds'
   AND temporality = 'delta';
-- Q1: Wider time range with filters.
SELECT locations, value, period, time_nanos, labels.comm FROM stacktraces
 WHERE time_nanos >= arrow_cast(0, 'Timestamp(Nanosecond, Some("UTC"))')
   AND time_nanos <= arrow_cast(250000000000, 'Timestamp(Nanosecond, Some("UTC"))')
   AND producer = 'parca_agent'
   AND sample_type = 'samples'
   AND sample_unit = 'count'
   AND period_type = 'cpu'
   AND period_unit = 'nanoseconds'
   AND temporality = 'delta';
-- Q2: Q1 + label equality filter.
SELECT locations, value, period, time_nanos, labels.comm FROM stacktraces
 WHERE time_nanos >= arrow_cast(0, 'Timestamp(Nanosecond, Some("UTC"))')
   AND time_nanos <= arrow_cast(250000000000, 'Timestamp(Nanosecond, Some("UTC"))')
   AND producer = 'parca_agent'
   AND sample_type = 'samples'
   AND sample_unit = 'count'
   AND period_type = 'cpu'
   AND period_unit = 'nanoseconds'
   AND temporality = 'delta'
   AND labels.comm = 'comm_0';
-- Q3: Q1 + label regex matcher.
SELECT locations, value, period, time_nanos, labels.comm FROM stacktraces
 WHERE time_nanos >= arrow_cast(0, 'Timestamp(Nanosecond, Some("UTC"))')
   AND time_nanos <= arrow_cast(250000000000, 'Timestamp(Nanosecond, Some("UTC"))')
   AND producer = 'parca_agent'
   AND sample_type = 'samples'
   AND sample_unit = 'count'
   AND period_type = 'cpu'
   AND period_unit = 'nanoseconds'
   AND temporality = 'delta'
   AND labels.comm ~ '^comm_0$';
-- Q4: Time-based aggregation.
SELECT date_bin(INTERVAL '1 second', time_nanos) AS timestamp_bucket,
       SUM(value) AS value, SUM(duration) AS duration
  FROM stacktraces
 WHERE time_nanos >= arrow_cast(0, 'Timestamp(Nanosecond, Some("UTC"))')
   AND time_nanos <= arrow_cast(250000000000, 'Timestamp(Nanosecond, Some("UTC"))')
   AND producer = 'parca_agent'
   AND sample_type = 'samples'
   AND sample_unit = 'count'
   AND period_type = 'cpu'
   AND period_unit = 'nanoseconds'
   AND temporality = 'delta'
 GROUP BY timestamp_bucket;
-- Q5: Q4 + label grouping.
SELECT date_bin(INTERVAL '1 second', time_nanos) AS timestamp_bucket,
       SUM(value) AS value, SUM(duration) AS duration,
       labels.comm
  FROM stacktraces
 WHERE time_nanos >= arrow_cast(0, 'Timestamp(Nanosecond, Some("UTC"))')
   AND time_nanos <= arrow_cast(250000000000, 'Timestamp(Nanosecond, Some("UTC"))')
   AND producer = 'parca_agent'
   AND sample_type = 'samples'
   AND sample_unit = 'count'
   AND period_type = 'cpu'
   AND period_unit = 'nanoseconds'
   AND temporality = 'delta'
 GROUP BY timestamp_bucket, labels.comm;
-- Q6: Distinct label sets.
SELECT DISTINCT labels FROM stacktraces
 WHERE time_nanos >= arrow_cast(0, 'Timestamp(Nanosecond, Some("UTC"))')
   AND time_nanos <= arrow_cast(250000000000, 'Timestamp(Nanosecond, Some("UTC"))')
   AND producer = 'parca_agent'
   AND sample_type = 'samples'
   AND sample_unit = 'count'
   AND period_type = 'cpu'
   AND period_unit = 'nanoseconds'
   AND temporality = 'delta';
-- Q7: Distinct single label values.
SELECT DISTINCT labels.comm FROM stacktraces
 WHERE time_nanos >= arrow_cast(0, 'Timestamp(Nanosecond, Some("UTC"))')
   AND time_nanos <= arrow_cast(250000000000, 'Timestamp(Nanosecond, Some("UTC"))')
   AND producer = 'parca_agent'
   AND sample_type = 'samples'
   AND sample_unit = 'count'
   AND period_type = 'cpu'
   AND period_unit = 'nanoseconds'
   AND temporality = 'delta'
   AND labels.comm IS NOT NULL;
-- Q8: Distinct mapping files via unnest.
SELECT DISTINCT location.mapping_file
  FROM (SELECT unnest(locations) AS location FROM stacktraces
 WHERE time_nanos >= arrow_cast(0, 'Timestamp(Nanosecond, Some("UTC"))')
   AND time_nanos <= arrow_cast(250000000000, 'Timestamp(Nanosecond, Some("UTC"))')
   AND producer = 'parca_agent'
   AND sample_type = 'samples'
   AND sample_unit = 'count'
   AND period_type = 'cpu'
   AND period_unit = 'nanoseconds'
   AND temporality = 'delta');
-- Q9: Distinct profile types.
SELECT DISTINCT producer, sample_type, sample_unit, period_type, period_unit, temporality
  FROM stacktraces
 WHERE time_nanos >= arrow_cast(0, 'Timestamp(Nanosecond, Some("UTC"))')
   AND time_nanos <= arrow_cast(250000000000, 'Timestamp(Nanosecond, Some("UTC"))');
