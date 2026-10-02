-- single-groupby-1-1-1
SELECT date_trunc('minute', "time") AS minute,
max(usage_user) as max_usage_user
FROM cpu
WHERE hostname IN ('host_249') AND "time" >= TIMESTAMP '2016-01-03 12:26:46.646325' AND "time" < TIMESTAMP '2016-01-03 13:26:46.646325'
GROUP BY minute ORDER BY minute ASC;
-- single-groupby-1-1-12
SELECT date_trunc('minute', "time") AS minute,
max(usage_user) as max_usage_user
FROM cpu
WHERE hostname IN ('host_249') AND "time" >= TIMESTAMP '2016-01-02 11:22:40.646325' AND "time" < TIMESTAMP '2016-01-02 23:22:40.646325'
GROUP BY minute ORDER BY minute ASC;
-- single-groupby-1-8-1
SELECT date_trunc('minute', "time") AS minute,
max(usage_user) as max_usage_user
FROM cpu
WHERE hostname IN ('host_249','host_1403','host_1435','host_3539','host_3639','host_3075','host_815','host_2121') AND "time" >= TIMESTAMP '2016-01-03 12:26:46.646325' AND "time" < TIMESTAMP '2016-01-03 13:26:46.646325'
GROUP BY minute ORDER BY minute ASC;
-- single-groupby-5-1-1
SELECT date_trunc('minute', "time") AS minute,
max(usage_user) as max_usage_user, max(usage_system) as max_usage_system, max(usage_idle) as max_usage_idle, max(usage_nice) as max_usage_nice, max(usage_iowait) as max_usage_iowait
FROM cpu
WHERE hostname IN ('host_249') AND "time" >= TIMESTAMP '2016-01-03 12:26:46.646325' AND "time" < TIMESTAMP '2016-01-03 13:26:46.646325'
GROUP BY minute ORDER BY minute ASC;
-- single-groupby-5-1-12
SELECT date_trunc('minute', "time") AS minute,
max(usage_user) as max_usage_user, max(usage_system) as max_usage_system, max(usage_idle) as max_usage_idle, max(usage_nice) as max_usage_nice, max(usage_iowait) as max_usage_iowait
FROM cpu
WHERE hostname IN ('host_249') AND "time" >= TIMESTAMP '2016-01-02 11:22:40.646325' AND "time" < TIMESTAMP '2016-01-02 23:22:40.646325'
GROUP BY minute ORDER BY minute ASC;
-- single-groupby-5-8-1
SELECT date_trunc('minute', "time") AS minute,
max(usage_user) as max_usage_user, max(usage_system) as max_usage_system, max(usage_idle) as max_usage_idle, max(usage_nice) as max_usage_nice, max(usage_iowait) as max_usage_iowait
FROM cpu
WHERE hostname IN ('host_249','host_1403','host_1435','host_3539','host_3639','host_3075','host_815','host_2121') AND "time" >= TIMESTAMP '2016-01-03 12:26:46.646325' AND "time" < TIMESTAMP '2016-01-03 13:26:46.646325'
GROUP BY minute ORDER BY minute ASC;
-- cpu-max-all-1
SELECT date_trunc('hour', "time") AS hour,
max(usage_user) as max_usage_user, max(usage_system) as max_usage_system, max(usage_idle) as max_usage_idle, max(usage_nice) as max_usage_nice, max(usage_iowait) as max_usage_iowait, max(usage_irq) as max_usage_irq, max(usage_softirq) as max_usage_softirq, max(usage_steal) as max_usage_steal, max(usage_guest) as max_usage_guest, max(usage_guest_nice) as max_usage_guest_nice
FROM cpu
WHERE hostname IN ('host_249') AND "time" >= TIMESTAMP '2016-01-01 11:48:31.646325' AND "time" < TIMESTAMP '2016-01-01 19:48:31.646325'
GROUP BY hour ORDER BY hour;
-- cpu-max-all-8
SELECT date_trunc('hour', "time") AS hour,
max(usage_user) as max_usage_user, max(usage_system) as max_usage_system, max(usage_idle) as max_usage_idle, max(usage_nice) as max_usage_nice, max(usage_iowait) as max_usage_iowait, max(usage_irq) as max_usage_irq, max(usage_softirq) as max_usage_softirq, max(usage_steal) as max_usage_steal, max(usage_guest) as max_usage_guest, max(usage_guest_nice) as max_usage_guest_nice
FROM cpu
WHERE hostname IN ('host_249','host_1403','host_1435','host_3539','host_3639','host_3075','host_815','host_2121') AND "time" >= TIMESTAMP '2016-01-01 11:48:31.646325' AND "time" < TIMESTAMP '2016-01-01 19:48:31.646325'
GROUP BY hour ORDER BY hour;
-- double-groupby-1
WITH cpu_avg AS (
SELECT date_trunc('hour', "time") as hour, hostname,
avg(usage_user) as mean_usage_user
FROM cpu
WHERE "time" >= TIMESTAMP '2016-01-02 11:22:40.646325' AND "time" < TIMESTAMP '2016-01-02 23:22:40.646325'
GROUP BY 1, 2
)
SELECT hour, hostname, mean_usage_user
FROM cpu_avg
ORDER BY hour, hostname;
-- double-groupby-5
WITH cpu_avg AS (
SELECT date_trunc('hour', "time") as hour, hostname,
avg(usage_user) as mean_usage_user, avg(usage_system) as mean_usage_system, avg(usage_idle) as mean_usage_idle, avg(usage_nice) as mean_usage_nice, avg(usage_iowait) as mean_usage_iowait
FROM cpu
WHERE "time" >= TIMESTAMP '2016-01-02 11:22:40.646325' AND "time" < TIMESTAMP '2016-01-02 23:22:40.646325'
GROUP BY 1, 2
)
SELECT hour, hostname, mean_usage_user, mean_usage_system, mean_usage_idle, mean_usage_nice, mean_usage_iowait
FROM cpu_avg
ORDER BY hour, hostname;
-- double-groupby-all
WITH cpu_avg AS (
SELECT date_trunc('hour', "time") as hour, hostname,
avg(usage_user) as mean_usage_user, avg(usage_system) as mean_usage_system, avg(usage_idle) as mean_usage_idle, avg(usage_nice) as mean_usage_nice, avg(usage_iowait) as mean_usage_iowait, avg(usage_irq) as mean_usage_irq, avg(usage_softirq) as mean_usage_softirq, avg(usage_steal) as mean_usage_steal, avg(usage_guest) as mean_usage_guest, avg(usage_guest_nice) as mean_usage_guest_nice
FROM cpu
WHERE "time" >= TIMESTAMP '2016-01-02 11:22:40.646325' AND "time" < TIMESTAMP '2016-01-02 23:22:40.646325'
GROUP BY 1, 2
)
SELECT hour, hostname, mean_usage_user, mean_usage_system, mean_usage_idle, mean_usage_nice, mean_usage_iowait, mean_usage_irq, mean_usage_softirq, mean_usage_steal, mean_usage_guest, mean_usage_guest_nice
FROM cpu_avg
ORDER BY hour, hostname;
-- high-cpu-all
SELECT * FROM cpu WHERE usage_user > 90.0 and "time" >= TIMESTAMP '2016-01-02 11:22:40.646325' AND "time" < TIMESTAMP '2016-01-02 23:22:40.646325';
-- high-cpu-1
SELECT * FROM cpu WHERE usage_user > 90.0 and "time" >= TIMESTAMP '2016-01-02 23:35:31.138978' AND "time" < TIMESTAMP '2016-01-03 11:35:31.138978' AND hostname IN ('host_1035');
-- lastpoint
SELECT DISTINCT ON (hostname) * FROM cpu ORDER BY hostname, "time" DESC;
-- groupby-orderby-limit
SELECT date_trunc('minute', "time") AS minute, max(usage_user)
FROM cpu
WHERE "time" < TIMESTAMP '2016-01-03 13:26:46.646325'
GROUP BY minute
ORDER BY minute DESC
LIMIT 5;
