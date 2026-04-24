import http from "http";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import zlib from "zlib";
import readline from "readline";
import { Readable } from "stream";
import { LTTB } from "downsample";
import { QUERY_SUITES, FAN_OUT_GROUPS, ENGINE_RENAMES } from "./src/config.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Configuration
const PORT = process.env.PORT || 3000;
const DATA_URL =
  process.env.DATA_URL ||
  "https://vortex-ci-benchmark-results.s3.amazonaws.com/data.json.gz";
const COMMITS_URL =
  process.env.COMMITS_URL ||
  "https://vortex-ci-benchmark-results.s3.amazonaws.com/commits.json";
const REFRESH_INTERVAL = process.env.REFRESH_INTERVAL || 5 * 60 * 1000;
const MAX_POINTS = 200;
const USE_LOCAL_DATA = process.env.USE_LOCAL_DATA === "true";

// Benchmark groups: non-query groups + simple suites + fan-out suites
const GROUPS = [
  "Random Access",
  "Compression",
  "Compression Size",
  ...QUERY_SUITES.filter((s) => !s.skip && !s.fanOut).map((s) => s.displayName),
  ...FAN_OUT_GROUPS,
];

const MIME = {
  ".html": "text/html",
  ".js": "application/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".webmanifest": "application/manifest+json",
};

let store = {
  commits: [],
  groups: {},
  metadata: null,
  downsampled: {},
  lastUpdated: null,
};

// Utilities
const rename = (s) => ENGINE_RENAMES[s.toLowerCase()] || ENGINE_RENAMES[s] || s;
const geoMean = (arr) =>
  arr.length
    ? Math.pow(
        arr.reduce((a, v) => a * v, 1),
        1 / arr.length,
      )
    : null;

// Categorize benchmarks based on name patterns and metadata
function getGroup(benchmark) {
  const name = benchmark.name;
  const lower = name.toLowerCase();

  // Random Access: "random-access/..." or "random access/..."
  if (
    lower.startsWith("random-access/") ||
    lower.startsWith("random access/")
  ) {
    return "Random Access";
  }

  // Compression Size: size measurements
  if (
    lower.startsWith("vortex size/") ||
    lower.startsWith("vortex-file-compressed size/") ||
    lower.startsWith("parquet size/") ||
    lower.startsWith("lance size/") ||
    lower.includes(":raw size/") ||
    lower.includes(":parquet-zstd size/") ||
    lower.includes(":lance size/")
  ) {
    return "Compression Size";
  }

  // Compression: compress/decompress time and ratio measurements
  if (
    lower.startsWith("compress time/") ||
    lower.startsWith("decompress time/") ||
    lower.startsWith("parquet_rs-zstd compress") ||
    lower.startsWith("parquet_rs-zstd decompress") ||
    lower.startsWith("lance compress") ||
    lower.startsWith("lance decompress") ||
    lower.startsWith("vortex:lance ratio") ||
    lower.startsWith("vortex:parquet-zstd ratio") ||
    lower.startsWith("vortex:raw ratio")
  ) {
    return "Compression";
  }

  // SQL query suites: match "{prefix}_q..." or "{prefix}/..."
  for (const suite of QUERY_SUITES) {
    if (
      !lower.startsWith(suite.prefix + "_q") &&
      !lower.startsWith(suite.prefix + "/")
    )
      continue;
    if (suite.skip) return null;
    if (!suite.fanOut) return suite.displayName;
    // Fan-out suites: expand by storage and scale factor
    const storage = benchmark.storage?.toUpperCase() === "S3" ? "S3" : "NVMe";
    const rawSf = benchmark.dataset?.[suite.datasetKey]?.scale_factor;
    const sf = rawSf ? Math.round(parseFloat(rawSf)) : 1;
    return `${suite.displayName} (${storage}) (SF=${sf})`;
  }

  return null;
}

// Format query name for display: "{prefix}_q00" -> "{QUERY_PREFIX} Q0"
function formatQuery(q) {
  const lower = q.toLowerCase();
  for (const suite of QUERY_SUITES) {
    if (suite.skip) continue;
    const m = lower.match(new RegExp(`^${suite.prefix}[_ ]?q(\\d+)`, "i"));
    if (m) return `${suite.queryPrefix} Q${parseInt(m[1], 10)}`;
  }
  return q.toUpperCase().replace(/[_-]/g, " ");
}

function normalizeChartName(group, chartName) {
  if (group === "Compression Size" && chartName === "VORTEX FILE COMPRESSED SIZE") {
    return "VORTEX SIZE";
  }
  return chartName;
}

// LTTB downsampling
function lttbIndices(seriesMap, target) {
  const keys = [...seriesMap.keys()];
  if (!keys.length) return [];
  const len = seriesMap.get(keys[0])?.length || 0;
  if (len <= target) return [...Array(len).keys()];

  const avg = Array(len);
  for (let i = 0; i < len; i++) {
    let sum = 0,
      n = 0;
    for (const arr of seriesMap.values()) {
      const v = arr[i]?.value ?? arr[i];
      if (v != null && !isNaN(v)) {
        sum += v;
        n++;
      }
    }
    avg[i] = [i, n ? sum / n : 0];
  }

  const idx = LTTB(avg, target).map((p) => Math.round(p[0]));
  if (!idx.includes(0)) idx.unshift(0);
  if (!idx.includes(len - 1)) idx.push(len - 1);
  return idx.sort((a, b) => a - b);
}

function downsample(data, factor) {
  const target = Math.ceil(data.commits.length / factor);
  if (target >= data.commits.length) return data;

  const idx = lttbIndices(data.series, target);
  const series = new Map();
  for (const [k, v] of data.series)
    series.set(
      k,
      idx.map((i) => v[i]),
    );

  return {
    ...data,
    commits: idx.map((i) => data.commits[i]),
    series,
    originalLength: data.commits.length,
  };
}

// Data fetching — streams response body directly instead of buffering
async function fetchJsonl(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Fetch failed: ${url} ${res.status}`);
  return new Promise((resolve, reject) => {
    const results = [];
    const rl = readline.createInterface({
      input: Readable.fromWeb(res.body),
      crlfDelay: Infinity,
    });
    rl.on("line", (l) => {
      if (l.trim())
        try {
          results.push(JSON.parse(l));
        } catch {}
    });
    rl.on("close", () => resolve(results));
    rl.on("error", reject);
  });
}

function readLocalJsonl(fp) {
  return new Promise((resolve, reject) => {
    const results = [];
    const rl = readline.createInterface({
      input: fs.createReadStream(fp),
      crlfDelay: Infinity,
    });
    rl.on("line", (l) => {
      if (l.trim())
        try {
          results.push(JSON.parse(l));
        } catch {}
    });
    rl.on("close", () => resolve(results));
    rl.on("error", reject);
  });
}

// Stream benchmark data record-by-record without buffering the entire dataset
async function forEachBenchmark(callback) {
  let stream;
  if (USE_LOCAL_DATA) {
    stream = fs.createReadStream(path.join(__dirname, "sample/data.json"));
  } else {
    const res = await fetch(DATA_URL);
    if (!res.ok) throw new Error(`Fetch failed: ${DATA_URL} ${res.status}`);
    stream = Readable.fromWeb(res.body).pipe(zlib.createGunzip());
  }
  return new Promise((resolve, reject) => {
    const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });
    rl.on("line", (l) => {
      if (l.trim())
        try {
          callback(JSON.parse(l));
        } catch {}
    });
    rl.on("close", resolve);
    rl.on("error", reject);
  });
}

// Main data processing
async function refresh() {
  console.log("Refreshing data...");
  const t0 = Date.now();

  try {
    // Load commits first (small dataset, must be fully in memory for indexing)
    const commitsArr = USE_LOCAL_DATA
      ? await readLocalJsonl(path.join(__dirname, "sample/commits.json"))
      : await fetchJsonl(COMMITS_URL);

    // Build commit index (O(1) lookup)
    const commitMap = new Map(commitsArr.map((c) => [c.id, c]));
    const commits = commitsArr.sort(
      (a, b) => new Date(a.timestamp) - new Date(b.timestamp),
    );
    const commitIdx = new Map(commits.map((c, i) => [c.id, i]));

    const groups = Object.fromEntries(GROUPS.map((g) => [g, new Map()]));
    let missing = 0;
    let benchmarkCount = 0;
    const uncategorized = new Set();

    // Stream benchmarks one record at a time to avoid loading all into memory
    await forEachBenchmark((b) => {
      benchmarkCount++;
      const commit = b.commit || commitMap.get(b.commit_id);
      if (!commit) {
        missing++;
        return;
      }

      const group = getGroup(b);
      if (!group) {
        uncategorized.add(b.name.split("/")[0]);
        return;
      }
      if (!groups[group]) return;

      // Random access names have the form: random-access/{dataset}/{pattern}/{format}
      // Historical random access names: random-access/{format}
      // Other benchmarks use: {query}/{series}
      let seriesName, chartName;
      const parts = b.name.split("/");
      if (group === "Random Access" && parts.length === 4) {
        chartName = `${parts[1]}/${parts[2]}`.toUpperCase().replace(/[_-]/g, " ");
        seriesName = rename(parts[3] || "default");
      } else if (group === "Random Access" && parts.length === 2) {
        chartName = "RANDOM ACCESS";
        seriesName = rename(parts[1] || "default");
      } else {
        seriesName = rename(parts[1] || "default");
        chartName = formatQuery(parts[0]);
      }
      chartName = normalizeChartName(group, chartName);
      if (chartName.includes("PARQUET-UNC")) return;

      // Skip throughput metrics (keep only time/size)
      if (b.name.includes(" throughput")) return;

      let unit = b.unit;
      if (!unit) {
        if (b.name.toLowerCase().includes(" size/")) unit = "bytes";
        else if (b.name.toLowerCase().includes(" ratio ")) unit = "ratio";
        else unit = "ns";
      }

      const sortPos = parts[0].match(/q(\d+)$/i)?.[1]
        ? parseInt(RegExp.$1, 10)
        : 0;
      const idx = commitIdx.get(commit.id);
      if (idx === undefined) return;

      let chart = groups[group].get(chartName);
      if (!chart) {
        let displayUnit = unit;
        if (unit === "ns") displayUnit = "ms/iter";
        else if (unit === "bytes") displayUnit = "MiB";
        chart = {
          sort_position: sortPos,
          commits,
          unit: displayUnit,
          series: new Map(),
        };
        groups[group].set(chartName, chart);
      }

      if (!chart.series.has(seriesName)) {
        chart.series.set(seriesName, Array(commits.length).fill(null));
      }

      // Convert values: ns -> ms, bytes -> MiB
      let val = b.value;
      if (unit === "ns" && typeof val === "number") {
        val = val / 1e6; // ns to ms
      } else if (unit === "bytes" && typeof val === "number") {
        val = val / (1024 * 1024); // bytes to MiB
      }

      chart.series.get(seriesName)[idx] = { value: val };
    });

    console.log(
      `Processed ${benchmarkCount} benchmarks, ${commitsArr.length} commits`,
    );

    // Log uncategorized benchmarks for debugging
    if (uncategorized.size > 0) {
      console.log(
        `Uncategorized benchmark prefixes (${uncategorized.size}):`,
        [...uncategorized].slice(0, 20).join(", "),
      );
    }

    // Trim leading empty commits
    let firstIdx = commits.length;
    for (const gc of Object.values(groups)) {
      for (const cd of gc.values()) {
        for (const sd of cd.series.values()) {
          const i = sd.findIndex((d) => d !== null);
          if (i !== -1 && i < firstIdx) firstIdx = i;
        }
      }
    }

    if (firstIdx > 0 && firstIdx < commits.length) {
      console.log(`Trimming ${firstIdx} empty commits`);
      commits.splice(0, firstIdx);
      for (const gc of Object.values(groups)) {
        for (const cd of gc.values()) {
          cd.commits = commits;
          for (const [k, v] of cd.series) cd.series.set(k, v.slice(firstIdx));
        }
      }
    }

    // Sort charts within groups
    for (const gc of Object.values(groups)) {
      const sorted = [...gc.entries()].sort(
        (a, b) =>
          a[1].sort_position - b[1].sort_position || a[0].localeCompare(b[0]),
      );
      gc.clear();
      for (const [k, v] of sorted) gc.set(k, v);
    }

    // Precompute downsampled versions
    const downsampled = {};
    for (const [gn, gc] of Object.entries(groups)) {
      downsampled[gn] = {};
      for (const [cn, cd] of gc) {
        downsampled[gn][cn] = {
          "1x": cd,
          "2x": downsample(cd, 2),
          "4x": downsample(cd, 4),
          "8x": downsample(cd, 8),
        };
      }
    }

    // Count charts per group for logging
    const groupCounts = Object.entries(groups)
      .map(([n, g]) => `${n}: ${g.size}`)
      .filter((s) => !s.endsWith(": 0"));
    console.log("Charts per group:", groupCounts.join(", "));

    store = {
      commits,
      groups,
      metadata: buildMeta(groups, commits),
      downsampled,
      lastUpdated: new Date().toISOString(),
    };
    console.log(
      `Refresh done in ${Date.now() - t0}ms (${missing} missing commits)`,
    );
  } catch (e) {
    console.error("Refresh error:", e);
  }
}

// Summary calculations
function latestIdx(chart) {
  for (let i = chart.commits.length - 1; i >= 0; i--) {
    for (const s of chart.series.values()) if (s[i]?.value != null) return i;
  }
  return -1;
}

function calcSummary(name, charts) {
  if (name === "Random Access") {
    for (const q of charts.values()) {
      const i = latestIdx(q);
      if (i === -1) continue;
      const vals = new Map();
      for (const [n, d] of q.series)
        if (d[i]?.value != null) vals.set(n, d[i].value);
      if (!vals.size) continue;
      const min = Math.min(...vals.values());
      return {
        type: "randomAccess",
        title: "Random Access Performance",
        rankings: [...vals]
          .map(([n, t]) => ({ name: n, time: t, ratio: t / min }))
          .sort((a, b) => a.time - b.time),
        explanation: "Random access time | Ratio to fastest (lower is better)",
      };
    }
    return null;
  }

  if (name === "Compression") {
    const cc = charts.get("VORTEX:PARQUET ZSTD RATIO COMPRESS TIME");
    const dc = charts.get("VORTEX:PARQUET ZSTD RATIO DECOMPRESS TIME");
    if (!cc && !dc) return null;
    const i = latestIdx(cc || dc);
    if (i === -1) return null;
    const collect = (c) =>
      c
        ? [...c.series]
            .filter(([n]) => !n.toLowerCase().includes("wide table"))
            .map(([, d]) => d[i]?.value)
            .filter((v) => v > 0)
            .map((v) => 1 / v)
        : [];
    return {
      type: "compression",
      title: "Compression Throughput vs Parquet",
      compressRatio: geoMean(collect(cc)),
      decompressRatio: geoMean(collect(dc)),
      datasetCount: collect(cc).length,
      explanation:
        "Inverse geomean of Vortex/Parquet ratios (higher is better)",
    };
  }

  if (name === "Compression Size") {
    const c = charts.get("VORTEX:PARQUET ZSTD SIZE");
    if (!c) return null;
    const i = latestIdx(c);
    if (i === -1) return null;
    const ratios = [...c.series]
      .filter(([n]) => !n.toLowerCase().includes("wide table"))
      .map(([, d]) => d[i]?.value)
      .filter((v) => v > 0);
    return ratios.length
      ? {
          type: "compressionSize",
          title: "Compression Size Summary",
          minRatio: Math.min(...ratios),
          meanRatio: geoMean(ratios),
          maxRatio: Math.max(...ratios),
          datasetCount: ratios.length,
          explanation:
            "Geomean of Vortex/Parquet size ratios (lower is better)",
        }
      : null;
  }

  if (
    QUERY_SUITES.some(
      (s) =>
        !s.skip &&
        (name === s.displayName || name.startsWith(s.displayName + " ")),
    )
  ) {
    const all = new Map();
    for (const q of charts.values())
      for (const n of q.series.keys()) if (!all.has(n)) all.set(n, new Map());
    for (const [qn, qd] of charts) {
      for (const [sn, sd] of qd.series) {
        for (let i = sd.length - 1; i >= 0; i--) {
          if (sd[i]?.value != null) {
            all.get(sn).set(qn, sd[i].value);
            break;
          }
        }
      }
    }
    if (!all.size) return null;

    const scores = new Map();
    for (const [sn, qr] of all) {
      let total = 0,
        max = 0;
      for (const v of qr.values()) {
        total += v;
        max = Math.max(max, v);
      }
      const penalty = Math.max(300000, max) * 2;
      const ratios = [];
      for (const qn of charts.keys()) {
        let base = Infinity;
        for (const m of all.values())
          if (m.has(qn)) base = Math.min(base, m.get(qn));
        if (base < Infinity)
          ratios.push((10 + (qr.get(qn) ?? penalty)) / (10 + base));
      }
      if (ratios.length)
        scores.set(sn, { score: geoMean(ratios), totalRuntime: total });
    }

    return scores.size
      ? {
          type: "queryBenchmark",
          title: "Performance Summary",
          rankings: [...scores]
            .map(([n, d]) => ({ name: n, ...d }))
            .sort((a, b) => a.score - b.score),
          explanation:
            "Geomean of query time ratio to fastest (lower is better)",
        }
      : null;
  }
  return null;
}

function buildMeta(groups, commits) {
  const meta = {};
  for (const [gn, gc] of Object.entries(groups)) {
    const charts = [...gc].map(([cn, cd]) => {
      const latest = {};
      for (const [sn, sd] of cd.series) {
        for (let i = sd.length - 1; i >= 0; i--)
          if (sd[i]?.value != null) {
            latest[sn] = sd[i].value;
            break;
          }
      }
      return {
        name: cn,
        unit: cd.unit,
        series: [...cd.series.keys()],
        sortPosition: cd.sort_position,
        totalPoints: cd.commits.length,
        latestValues: latest,
      };
    });
    meta[gn] = {
      charts,
      totalCharts: charts.length,
      hasData: charts.length > 0,
      summary: calcSummary(gn, gc),
    };
  }
  return {
    groups: meta,
    totalCommits: commits.length,
    commits: commits.map((c) => ({
      id: c.id,
      message: c.message?.split("\n")[0] || "",
      timestamp: c.timestamp,
      author: c.author?.name || "Unknown",
    })),
    lastUpdated: new Date().toISOString(),
  };
}

// HTTP handlers
const json = (res, code, data) => {
  res.writeHead(code, {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
  });
  res.end(JSON.stringify(data));
};

function serveFile(res, fp) {
  fs.readFile(fp, (err, data) => {
    if (err) {
      res.writeHead(err.code === "ENOENT" ? 404 : 500);
      return res.end(err.code === "ENOENT" ? "Not Found" : "Error");
    }
    const ext = path.extname(fp).toLowerCase();
    const hdrs = { "Content-Type": MIME[ext] || "application/octet-stream" };
    if (ext === ".js") {
      hdrs["Cache-Control"] = "no-cache";
      hdrs["Pragma"] = "no-cache";
    }
    res.writeHead(200, hdrs);
    res.end(data);
  });
}

function handleData(res, group, chart, start, end, last, startIdx, endIdx) {
  if (!store.downsampled) return json(res, 503, { error: "Loading" });
  const gd = store.downsampled[group];
  if (!gd) return json(res, 404, { error: "Group not found" });
  const cv = gd[chart];
  if (!cv) return json(res, 404, { error: "Chart not found" });

  const full = cv["1x"];
  const ts = (c) =>
    typeof c?.timestamp === "number"
      ? c.timestamp
      : new Date(c?.timestamp).getTime();

  let si = 0,
    ei = full.commits.length - 1;

  // Support "last=N" parameter to get the last N commits
  if (last && !start && !end && startIdx === null && endIdx === null) {
    const n = parseInt(last, 10);
    if (n > 0 && n < full.commits.length) {
      si = full.commits.length - n;
    }
  } else if (startIdx !== null || endIdx !== null) {
    // Support index-based range (startIdx, endIdx)
    if (startIdx !== null) si = Math.max(0, parseInt(startIdx, 10));
    if (endIdx !== null)
      ei = Math.min(full.commits.length - 1, parseInt(endIdx, 10));
  } else {
    // Timestamp-based range
    if (start) {
      const t = +start,
        i = full.commits.findIndex((c) => ts(c) >= t);
      if (i !== -1) si = i;
    }
    if (end) {
      const t = +end;
      for (let i = ei; i >= 0; i--)
        if (ts(full.commits[i]) <= t) {
          ei = i;
          break;
        }
    }
  }

  const len = ei - si + 1;
  const ver =
    len <= MAX_POINTS
      ? "1x"
      : len <= MAX_POINTS * 2
        ? "2x"
        : len <= MAX_POINTS * 4
          ? "4x"
          : "8x";
  const cd = cv[ver];
  const val = (d) => d?.value ?? (typeof d === "number" ? d : null);

  let commits, series;
  if (ver === "1x") {
    commits = full.commits.slice(si, ei + 1);
    series = Object.fromEntries(
      [...full.series].map(([n, d]) => [n, d.slice(si, ei + 1).map(val)]),
    );
  } else {
    const s = +ver[0],
      dsi = Math.floor(si / s),
      dei = Math.min(Math.ceil(ei / s), cd.commits.length - 1);
    commits = cd.commits.slice(dsi, dei + 1);
    series = Object.fromEntries(
      [...cd.series].map(([n, d]) => [n, d.slice(dsi, dei + 1).map(val)]),
    );
  }

  json(res, 200, {
    group,
    chart,
    unit: cd.unit,
    downsampleLevel: ver,
    originalLength: full.commits.length,
    requestedRange: { startIndex: si, endIndex: ei, length: len },
    commits: commits.map((c) => ({
      id: c.id,
      message: c.message?.split("\n")[0] || "",
      timestamp: c.timestamp,
      author: c.author?.name || "Unknown",
      url: c.url,
    })),
    series,
  });
}

const server = http.createServer((req, res) => {
  const [path_, qs] = req.url.split("?");
  const params = new URLSearchParams(qs || "");

  if (req.method === "OPTIONS") {
    res.writeHead(204, {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET",
      "Access-Control-Allow-Headers": "Content-Type",
    });
    return res.end();
  }

  if (path_ === "/api/metadata")
    return store.metadata
      ? json(res, 200, store.metadata)
      : json(res, 503, { error: "Loading" });

  if (path_.startsWith("/api/data/")) {
    const p = path_.slice(10).split("/");
    return handleData(
      res,
      decodeURIComponent(p[0] || ""),
      decodeURIComponent(p.slice(1).join("/") || ""),
      params.get("start"),
      params.get("end"),
      params.get("last"),
      params.has("startIdx") ? params.get("startIdx") : null,
      params.has("endIdx") ? params.get("endIdx") : null,
    );
  }

  const fp = path.join(__dirname, "dist", path_ === "/" ? "index.html" : path_);
  if (!fp.startsWith(__dirname) || fp.includes("/sample/")) {
    res.writeHead(403);
    return res.end("Forbidden");
  }
  serveFile(res, fp);
});

async function start() {
  console.log("Starting server...");
  await refresh();
  setInterval(refresh, REFRESH_INTERVAL);
  server.listen(PORT, () => console.log(`Server at http://localhost:${PORT}`));
}

start().catch(console.error);
