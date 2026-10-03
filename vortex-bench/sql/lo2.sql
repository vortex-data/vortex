-- Real PromQL queries translated to SQL. Each query cites its source and quotes the original
-- PromQL. Sources, pinned to the commits they were copied from:
--   node-mixin: prometheus/node_exporter docs/node-mixin at 60ce437b, rules.libsonnet and
--     alerts.libsonnet, with the mixin defaults job="node", rateInterval 5m, device!="" for disks
--     and device!~"lo|veth.+" for physical network interfaces.
--   awesome-prometheus-alerts: the host and hardware rules as copied into
--     VictoriaMetrics/prometheus-benchmark chart/files/alerts.yaml at d2365722.
--   Node Exporter Full: Grafana dashboard 1860, revision 101, with $node = node_exporter:9100,
--     $job = node and $__rate_interval = 1m.
--   prombench: prometheus/test-infra prombench/manifests/prombench/benchmark/8_loadgen.yaml at
--     6fe6de1f.
--
-- Translation rules:
--   `ts` is int64 milliseconds. Instant queries are evaluated at T = 1739746800000. The data is
--   on a one second grid, so an instant vector selector reads the sample at exactly T.
--   rate(x[w]) is (last - first) / elapsed seconds over the samples in (T - w, T], computed as
--   max - min because the inputs are counters. Prometheus's extrapolation to the window edges
--   and counter-reset handling are omitted. A series needs at least two samples, as in PromQL.
--   Windows used: 1m = ts > 1739746740000, 2m = ts > 1739746680000, 5m = ts > 1739746500000,
--   3h = ts > 1739736000000, each with ts <= 1739746800000.
--   Dashboard panels run as a range query over [1739745120000, 1739748720000] at a 1m step.
--   A gauge reads the sample at each step, `ts % 60000 = 0`. A rate at step t covers
--   (t - 1m, t], and t is `ts + (60000 - ts % 60000) % 60000`.
--   Series are keyed by the labels that vary within each metric, so constant labels such as
--   job and group are left out of the output. Regexes are anchored as in PromQL. Thresholds and
--   `and` become WHERE clauses and joins. Every query runs unchanged on DataFusion and DuckDB.

-- Q0: node-mixin recording rule instance:node_num_cpu:sum
-- PromQL: count without (cpu, mode) (node_cpu_seconds_total{job="node",mode="idle"})
SELECT labels.instance AS instance, count(*) AS value
  FROM samples
 WHERE labels.__name__ = 'node_cpu_seconds_total'
   AND labels.job = 'node'
   AND labels.mode = 'idle'
   AND ts = 1739746800000
 GROUP BY instance
 ORDER BY instance;

-- Q1: node-mixin recording rule instance:node_cpu_utilisation:rate5m
-- PromQL: 1 - avg without (cpu) (sum without (mode) (rate(node_cpu_seconds_total{job="node", mode=~"idle|iowait|steal"}[5m])))
WITH per_series AS (
  SELECT labels.instance AS instance, labels.cpu AS cpu, labels.mode AS mode,
         (max(value) - min(value)) / ((max(ts) - min(ts)) / 1000.0) AS r
    FROM samples
   WHERE labels.__name__ = 'node_cpu_seconds_total'
     AND labels.job = 'node'
     AND labels.mode ~ '^(idle|iowait|steal)$'
     AND ts > 1739746500000 AND ts <= 1739746800000
   GROUP BY instance, cpu, mode
  HAVING count(*) >= 2
), per_cpu AS (
  SELECT instance, cpu, sum(r) AS r
    FROM per_series
   GROUP BY instance, cpu
)
SELECT instance, 1 - avg(r) AS value
  FROM per_cpu
 GROUP BY instance
 ORDER BY instance;

-- Q2: node-mixin recording rule instance:node_load1_per_cpu:ratio, with the
-- instance:node_num_cpu:sum rule it divides by inlined.
-- PromQL: node_load1{job="node"} / instance:node_num_cpu:sum{job="node"}
WITH load1 AS (
  SELECT labels.instance AS instance, value
    FROM samples
   WHERE labels.__name__ = 'node_load1'
     AND labels.job = 'node'
     AND ts = 1739746800000
), cpus AS (
  SELECT labels.instance AS instance, count(*) AS n
    FROM samples
   WHERE labels.__name__ = 'node_cpu_seconds_total'
     AND labels.job = 'node'
     AND labels.mode = 'idle'
     AND ts = 1739746800000
   GROUP BY instance
)
SELECT load1.instance AS instance, load1.value / cpus.n AS value
  FROM load1
  JOIN cpus ON load1.instance = cpus.instance
 ORDER BY instance;

-- Q3: node-mixin recording rule instance:node_memory_utilisation:ratio. The `or` fallback is not
-- taken because node_memory_MemAvailable_bytes is present.
-- PromQL: 1 - ((node_memory_MemAvailable_bytes{job="node"} or (node_memory_Buffers_bytes{job="node"} + node_memory_Cached_bytes{job="node"} + node_memory_MemFree_bytes{job="node"} + node_memory_Slab_bytes{job="node"})) / node_memory_MemTotal_bytes{job="node"})
WITH avail AS (
  SELECT labels.instance AS instance, value
    FROM samples
   WHERE labels.__name__ = 'node_memory_MemAvailable_bytes'
     AND labels.job = 'node'
     AND ts = 1739746800000
), total AS (
  SELECT labels.instance AS instance, value
    FROM samples
   WHERE labels.__name__ = 'node_memory_MemTotal_bytes'
     AND labels.job = 'node'
     AND ts = 1739746800000
)
SELECT avail.instance AS instance, 1 - avail.value / total.value AS value
  FROM avail
  JOIN total ON avail.instance = total.instance
 ORDER BY instance;

-- Q4: node-mixin recording rule instance:node_vmstat_pgmajfault:rate5m
-- PromQL: rate(node_vmstat_pgmajfault{job="node"}[5m])
SELECT labels.instance AS instance,
       (max(value) - min(value)) / ((max(ts) - min(ts)) / 1000.0) AS value
  FROM samples
 WHERE labels.__name__ = 'node_vmstat_pgmajfault'
   AND labels.job = 'node'
   AND ts > 1739746500000 AND ts <= 1739746800000
 GROUP BY instance
HAVING count(*) >= 2
 ORDER BY instance;

-- Q5: node-mixin recording rule instance_device:node_disk_io_time_seconds:rate5m
-- PromQL: rate(node_disk_io_time_seconds_total{job="node", device!=""}[5m])
SELECT labels.instance AS instance, labels.device AS device,
       (max(value) - min(value)) / ((max(ts) - min(ts)) / 1000.0) AS value
  FROM samples
 WHERE labels.__name__ = 'node_disk_io_time_seconds_total'
   AND labels.job = 'node'
   AND labels.device <> ''
   AND ts > 1739746500000 AND ts <= 1739746800000
 GROUP BY instance, device
HAVING count(*) >= 2
 ORDER BY instance, device;

-- Q6: node-mixin recording rule instance_device:node_disk_io_time_weighted_seconds:rate5m
-- PromQL: rate(node_disk_io_time_weighted_seconds_total{job="node", device!=""}[5m])
SELECT labels.instance AS instance, labels.device AS device,
       (max(value) - min(value)) / ((max(ts) - min(ts)) / 1000.0) AS value
  FROM samples
 WHERE labels.__name__ = 'node_disk_io_time_weighted_seconds_total'
   AND labels.job = 'node'
   AND labels.device <> ''
   AND ts > 1739746500000 AND ts <= 1739746800000
 GROUP BY instance, device
HAVING count(*) >= 2
 ORDER BY instance, device;

-- Q7: node-mixin recording rule instance:node_network_receive_bytes_excluding_lo:rate5m
-- PromQL: sum without (device) (rate(node_network_receive_bytes_total{job="node", device!="lo"}[5m]))
WITH per_series AS (
  SELECT labels.instance AS instance, labels.device AS device,
         (max(value) - min(value)) / ((max(ts) - min(ts)) / 1000.0) AS r
    FROM samples
   WHERE labels.__name__ = 'node_network_receive_bytes_total'
     AND labels.job = 'node'
     AND labels.device <> 'lo'
     AND ts > 1739746500000 AND ts <= 1739746800000
   GROUP BY instance, device
  HAVING count(*) >= 2
)
SELECT instance, sum(r) AS value
  FROM per_series
 GROUP BY instance
 ORDER BY instance;

-- Q8: node-mixin recording rule instance:node_network_transmit_bytes_physical:rate5m
-- PromQL: sum without (device) (rate(node_network_transmit_bytes_total{job="node", device!~"lo|veth.+"}[5m]))
WITH per_series AS (
  SELECT labels.instance AS instance, labels.device AS device,
         (max(value) - min(value)) / ((max(ts) - min(ts)) / 1000.0) AS r
    FROM samples
   WHERE labels.__name__ = 'node_network_transmit_bytes_total'
     AND labels.job = 'node'
     AND labels.device !~ '^(lo|veth.+)$'
     AND ts > 1739746500000 AND ts <= 1739746800000
   GROUP BY instance, device
  HAVING count(*) >= 2
)
SELECT instance, sum(r) AS value
  FROM per_series
 GROUP BY instance
 ORDER BY instance;

-- Q9: node-mixin alert NodeHighNumberConntrackEntriesUsed
-- PromQL: (node_nf_conntrack_entries{job="node"} / node_nf_conntrack_entries_limit) > 0.75
WITH entries AS (
  SELECT labels.instance AS instance, value
    FROM samples
   WHERE labels.__name__ = 'node_nf_conntrack_entries'
     AND labels.job = 'node'
     AND ts = 1739746800000
), lim AS (
  SELECT labels.instance AS instance, value
    FROM samples
   WHERE labels.__name__ = 'node_nf_conntrack_entries_limit'
     AND ts = 1739746800000
)
SELECT entries.instance AS instance, entries.value / lim.value AS value
  FROM entries
  JOIN lim ON entries.instance = lim.instance
 WHERE entries.value / lim.value > 0.75
 ORDER BY instance;

-- Q10: awesome-prometheus-alerts HostOutOfMemory
-- PromQL: (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes < .10)
WITH avail AS (
  SELECT labels.instance AS instance, value
    FROM samples
   WHERE labels.__name__ = 'node_memory_MemAvailable_bytes'
     AND ts = 1739746800000
), total AS (
  SELECT labels.instance AS instance, value
    FROM samples
   WHERE labels.__name__ = 'node_memory_MemTotal_bytes'
     AND ts = 1739746800000
)
SELECT avail.instance AS instance, avail.value / total.value AS value
  FROM avail
  JOIN total ON avail.instance = total.instance
 WHERE avail.value / total.value < 0.10
 ORDER BY instance;

-- Q11: awesome-prometheus-alerts HostHighCpuLoad
-- PromQL: 1 - (avg without (cpu) (rate(node_cpu_seconds_total{mode="idle"}[5m]))) > .80
WITH per_series AS (
  SELECT labels.instance AS instance, labels.cpu AS cpu,
         (max(value) - min(value)) / ((max(ts) - min(ts)) / 1000.0) AS r
    FROM samples
   WHERE labels.__name__ = 'node_cpu_seconds_total'
     AND labels.mode = 'idle'
     AND ts > 1739746500000 AND ts <= 1739746800000
   GROUP BY instance, cpu
  HAVING count(*) >= 2
)
SELECT instance, 1 - avg(r) AS value
  FROM per_series
 GROUP BY instance
HAVING 1 - avg(r) > 0.80
 ORDER BY instance;

-- Q12: awesome-prometheus-alerts HostCpuHighIowait
-- PromQL: avg without (cpu) (rate(node_cpu_seconds_total{mode="iowait"}[5m])) > .10
WITH per_series AS (
  SELECT labels.instance AS instance, labels.cpu AS cpu,
         (max(value) - min(value)) / ((max(ts) - min(ts)) / 1000.0) AS r
    FROM samples
   WHERE labels.__name__ = 'node_cpu_seconds_total'
     AND labels.mode = 'iowait'
     AND ts > 1739746500000 AND ts <= 1739746800000
   GROUP BY instance, cpu
  HAVING count(*) >= 2
)
SELECT instance, avg(r) AS value
  FROM per_series
 GROUP BY instance
HAVING avg(r) > 0.10
 ORDER BY instance;

-- Q13: awesome-prometheus-alerts HostUnusualDiskIo
-- PromQL: rate(node_disk_io_time_seconds_total[5m]) > 0.8
SELECT labels.instance AS instance, labels.device AS device,
       (max(value) - min(value)) / ((max(ts) - min(ts)) / 1000.0) AS value
  FROM samples
 WHERE labels.__name__ = 'node_disk_io_time_seconds_total'
   AND ts > 1739746500000 AND ts <= 1739746800000
 GROUP BY instance, device
HAVING count(*) >= 2
   AND (max(value) - min(value)) / ((max(ts) - min(ts)) / 1000.0) > 0.8
 ORDER BY instance, device;

-- Q14: awesome-prometheus-alerts HostDiskMayFillIn24Hours. predict_linear is a least-squares fit
-- over the window, with time in seconds relative to T as Prometheus does.
-- PromQL: predict_linear(node_filesystem_avail_bytes{fstype!~"^(fuse.*|tmpfs|cifs|nfs)"}[3h], 86400) <= 0 and node_filesystem_avail_bytes > 0
WITH fit AS (
  SELECT labels.instance AS instance, labels.device AS device,
         labels.fstype AS fstype, labels.mountpoint AS mountpoint,
         regr_intercept(value, (ts - 1739746800000) / 1000.0)
           + regr_slope(value, (ts - 1739746800000) / 1000.0) * 86400 AS predicted
    FROM samples
   WHERE labels.__name__ = 'node_filesystem_avail_bytes'
     AND labels.fstype !~ '^(fuse.*|tmpfs|cifs|nfs)$'
     AND ts > 1739736000000 AND ts <= 1739746800000
   GROUP BY instance, device, fstype, mountpoint
  HAVING count(*) >= 2
), latest AS (
  SELECT labels.instance AS instance, labels.device AS device,
         labels.fstype AS fstype, labels.mountpoint AS mountpoint, value
    FROM samples
   WHERE labels.__name__ = 'node_filesystem_avail_bytes'
     AND ts = 1739746800000
)
SELECT fit.instance AS instance, fit.device AS device, fit.fstype AS fstype,
       fit.mountpoint AS mountpoint, fit.predicted AS value
  FROM fit
  JOIN latest
    ON fit.instance = latest.instance AND fit.device = latest.device
   AND fit.fstype = latest.fstype AND fit.mountpoint = latest.mountpoint
 WHERE fit.predicted <= 0
   AND latest.value > 0
 ORDER BY instance, device, mountpoint;

-- Q15: awesome-prometheus-alerts HostOutOfDiskSpace
-- PromQL: (node_filesystem_avail_bytes{fstype!~"^(fuse.*|tmpfs|cifs|nfs)"} / node_filesystem_size_bytes < .10 and on (instance, device, mountpoint) node_filesystem_readonly == 0)
WITH avail AS (
  SELECT labels.instance AS instance, labels.device AS device,
         labels.fstype AS fstype, labels.mountpoint AS mountpoint, value
    FROM samples
   WHERE labels.__name__ = 'node_filesystem_avail_bytes'
     AND labels.fstype !~ '^(fuse.*|tmpfs|cifs|nfs)$'
     AND ts = 1739746800000
), total AS (
  SELECT labels.instance AS instance, labels.device AS device,
         labels.fstype AS fstype, labels.mountpoint AS mountpoint, value
    FROM samples
   WHERE labels.__name__ = 'node_filesystem_size_bytes'
     AND ts = 1739746800000
), ro AS (
  SELECT labels.instance AS instance, labels.device AS device, labels.mountpoint AS mountpoint
    FROM samples
   WHERE labels.__name__ = 'node_filesystem_readonly'
     AND value = 0
     AND ts = 1739746800000
)
SELECT avail.instance AS instance, avail.device AS device, avail.fstype AS fstype,
       avail.mountpoint AS mountpoint, avail.value / total.value AS value
  FROM avail
  JOIN total
    ON avail.instance = total.instance AND avail.device = total.device
   AND avail.fstype = total.fstype AND avail.mountpoint = total.mountpoint
  JOIN ro
    ON avail.instance = ro.instance AND avail.device = ro.device
   AND avail.mountpoint = ro.mountpoint
 WHERE avail.value / total.value < 0.10
 ORDER BY instance, device, mountpoint;

-- Q16: awesome-prometheus-alerts HostNetworkReceiveErrors. The ratio test is written as
-- errs > 0.01 * packets, which gives PromQL's result without dividing by zero.
-- PromQL: (rate(node_network_receive_errs_total[2m]) / rate(node_network_receive_packets_total[2m]) > 0.01)
WITH errs AS (
  SELECT labels.instance AS instance, labels.device AS device,
         (max(value) - min(value)) / ((max(ts) - min(ts)) / 1000.0) AS r
    FROM samples
   WHERE labels.__name__ = 'node_network_receive_errs_total'
     AND ts > 1739746680000 AND ts <= 1739746800000
   GROUP BY instance, device
  HAVING count(*) >= 2
), packets AS (
  SELECT labels.instance AS instance, labels.device AS device,
         (max(value) - min(value)) / ((max(ts) - min(ts)) / 1000.0) AS r
    FROM samples
   WHERE labels.__name__ = 'node_network_receive_packets_total'
     AND ts > 1739746680000 AND ts <= 1739746800000
   GROUP BY instance, device
  HAVING count(*) >= 2
)
SELECT errs.instance AS instance, errs.device AS device,
       errs.r AS errs_rate, packets.r AS packets_rate
  FROM errs
  JOIN packets ON errs.instance = packets.instance AND errs.device = packets.device
 WHERE errs.r > 0.01 * packets.r
 ORDER BY instance, device;

-- Q17: Node Exporter Full, panel "CPU Busy", an instant gauge.
-- PromQL: 100 * (1 - avg(rate(node_cpu_seconds_total{mode="idle",instance="$node",job="$job"}[$__rate_interval])))
WITH per_series AS (
  SELECT labels.cpu AS cpu,
         (max(value) - min(value)) / ((max(ts) - min(ts)) / 1000.0) AS r
    FROM samples
   WHERE labels.__name__ = 'node_cpu_seconds_total'
     AND labels.mode = 'idle'
     AND labels.instance = 'node_exporter:9100'
     AND labels.job = 'node'
     AND ts > 1739746740000 AND ts <= 1739746800000
   GROUP BY cpu
  HAVING count(*) >= 2
)
SELECT 100 * (1 - avg(r)) AS value
  FROM per_series;

-- Q18: Node Exporter Full, panel "Sys Load", an instant gauge.
-- PromQL: scalar(node_load1{instance="$node",job="$job"}) * 100 / count(count(node_cpu_seconds_total{instance="$node",job="$job"}) by (cpu))
SELECT (SELECT value
          FROM samples
         WHERE labels.__name__ = 'node_load1'
           AND labels.instance = 'node_exporter:9100'
           AND labels.job = 'node'
           AND ts = 1739746800000) * 100
       / (SELECT count(DISTINCT labels.cpu)
            FROM samples
           WHERE labels.__name__ = 'node_cpu_seconds_total'
             AND labels.instance = 'node_exporter:9100'
             AND labels.job = 'node'
             AND ts = 1739746800000) AS value;

-- Q19: Node Exporter Full, panel "CPU Basic", the user series, as a range query.
-- PromQL: avg(rate(node_cpu_seconds_total{instance="$node",job="$job", mode="user"}[$__rate_interval]))
WITH per_series AS (
  SELECT ts + (60000 - ts % 60000) % 60000 AS step_ts, labels.cpu AS cpu,
         (max(value) - min(value)) / ((max(ts) - min(ts)) / 1000.0) AS r
    FROM samples
   WHERE labels.__name__ = 'node_cpu_seconds_total'
     AND labels.mode = 'user'
     AND labels.instance = 'node_exporter:9100'
     AND labels.job = 'node'
     AND ts > 1739745060000 AND ts <= 1739748720000
   GROUP BY step_ts, cpu
  HAVING count(*) >= 2
)
SELECT step_ts, avg(r) AS value
  FROM per_series
 GROUP BY step_ts
 ORDER BY step_ts;

-- Q20: Node Exporter Full, panel "Memory Basic", the RAM Used series, as a range query. The
-- operands share their labels, so vector matching is a pivot on the metric name.
-- PromQL: node_memory_MemTotal_bytes{instance="$node",job="$job"} - node_memory_MemFree_bytes{instance="$node",job="$job"} - (node_memory_Cached_bytes{instance="$node",job="$job"} + node_memory_Buffers_bytes{instance="$node",job="$job"} + node_memory_SReclaimable_bytes{instance="$node",job="$job"})
SELECT ts,
       sum(CASE WHEN labels.__name__ = 'node_memory_MemTotal_bytes' THEN value END)
     - sum(CASE WHEN labels.__name__ = 'node_memory_MemFree_bytes' THEN value END)
     - (sum(CASE WHEN labels.__name__ = 'node_memory_Cached_bytes' THEN value END)
        + sum(CASE WHEN labels.__name__ = 'node_memory_Buffers_bytes' THEN value END)
        + sum(CASE WHEN labels.__name__ = 'node_memory_SReclaimable_bytes' THEN value END)) AS value
  FROM samples
 WHERE labels.__name__ IN ('node_memory_MemTotal_bytes', 'node_memory_MemFree_bytes',
                           'node_memory_Cached_bytes', 'node_memory_Buffers_bytes',
                           'node_memory_SReclaimable_bytes')
   AND labels.instance = 'node_exporter:9100'
   AND labels.job = 'node'
   AND ts % 60000 = 0
   AND ts >= 1739745120000 AND ts <= 1739748720000
 GROUP BY ts
 ORDER BY ts;

-- Q21: Node Exporter Full, panel "Network Traffic Basic", the receive series, as a range query.
-- PromQL: rate(node_network_receive_bytes_total{instance="$node",job="$job"}[$__rate_interval])*8
SELECT ts + (60000 - ts % 60000) % 60000 AS step_ts, labels.device AS device,
       (max(value) - min(value)) / ((max(ts) - min(ts)) / 1000.0) * 8 AS value
  FROM samples
 WHERE labels.__name__ = 'node_network_receive_bytes_total'
   AND labels.instance = 'node_exporter:9100'
   AND labels.job = 'node'
   AND ts > 1739745060000 AND ts <= 1739748720000
 GROUP BY step_ts, device
HAVING count(*) >= 2
 ORDER BY step_ts, device;

-- Q22: Node Exporter Full, panel "Disk Space Used Basic", as a range query.
-- PromQL: ((node_filesystem_size_bytes{instance="$node", job="$job", device!~'rootfs'} - node_filesystem_avail_bytes{instance="$node", job="$job", device!~'rootfs'}) / node_filesystem_size_bytes{instance="$node", job="$job", device!~'rootfs'}) * 100
SELECT ts, labels.device AS device, labels.fstype AS fstype, labels.mountpoint AS mountpoint,
       (sum(CASE WHEN labels.__name__ = 'node_filesystem_size_bytes' THEN value END)
        - sum(CASE WHEN labels.__name__ = 'node_filesystem_avail_bytes' THEN value END))
       / sum(CASE WHEN labels.__name__ = 'node_filesystem_size_bytes' THEN value END) * 100 AS value
  FROM samples
 WHERE labels.__name__ IN ('node_filesystem_size_bytes', 'node_filesystem_avail_bytes')
   AND labels.instance = 'node_exporter:9100'
   AND labels.job = 'node'
   AND labels.device !~ '^(rootfs)$'
   AND ts % 60000 = 0
   AND ts >= 1739745120000 AND ts <= 1739748720000
 GROUP BY ts, device, fstype, mountpoint
 ORDER BY ts, device, mountpoint;

-- Q23: prombench aggr_instant
-- PromQL: sum by(image) (container_memory_rss)
SELECT labels.image AS image, sum(value) AS value
  FROM samples
 WHERE labels.__name__ = 'container_memory_rss'
   AND ts = 1739746800000
 GROUP BY image
 ORDER BY image;

-- Q24: prombench aggr_instant
-- PromQL: sum by(instance) (rate(node_cpu_seconds_total{mode!="idle"}[5m]))
WITH per_series AS (
  SELECT labels.instance AS instance, labels.cpu AS cpu, labels.mode AS mode,
         (max(value) - min(value)) / ((max(ts) - min(ts)) / 1000.0) AS r
    FROM samples
   WHERE labels.__name__ = 'node_cpu_seconds_total'
     AND labels.mode <> 'idle'
     AND ts > 1739746500000 AND ts <= 1739746800000
   GROUP BY instance, cpu, mode
  HAVING count(*) >= 2
)
SELECT instance, sum(r) AS value
  FROM per_series
 GROUP BY instance
 ORDER BY instance;

-- Q25: prombench simple_range, at its 15s step over the same hour as the dashboard panels.
-- PromQL: go_goroutines
SELECT ts, labels.job AS job, labels.instance AS instance, value
  FROM samples
 WHERE labels.__name__ = 'go_goroutines'
   AND ts % 15000 = 0
   AND ts >= 1739745120000 AND ts <= 1739748720000
 ORDER BY ts, job, instance;

-- Q26: prombench topk
-- PromQL: topk(20, sum(rate(go_gc_duration_seconds_count[5m])) by (instance, job))
WITH per_series AS (
  SELECT labels.instance AS instance, labels.job AS job,
         (max(value) - min(value)) / ((max(ts) - min(ts)) / 1000.0) AS r
    FROM samples
   WHERE labels.__name__ = 'go_gc_duration_seconds_count'
     AND ts > 1739746500000 AND ts <= 1739746800000
   GROUP BY instance, job
  HAVING count(*) >= 2
)
SELECT instance, job, sum(r) AS value
  FROM per_series
 GROUP BY instance, job
 ORDER BY value DESC, instance, job
 LIMIT 20;

-- Q27: prombench arithmetic_operation
-- PromQL: rate(go_memstats_frees_total[5m]) * 60
SELECT labels.job AS job, labels.instance AS instance,
       (max(value) - min(value)) / ((max(ts) - min(ts)) / 1000.0) * 60 AS value
  FROM samples
 WHERE labels.__name__ = 'go_memstats_frees_total'
   AND ts > 1739746500000 AND ts <= 1739746800000
 GROUP BY job, instance
HAVING count(*) >= 2
 ORDER BY job, instance;
