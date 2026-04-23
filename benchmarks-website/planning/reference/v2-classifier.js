
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
