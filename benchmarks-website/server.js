import http from "http";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { BenchmarkStore } from "./duckdb-store.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = process.env.PORT || 3000;
const DATA_URL =
  process.env.DATA_URL ||
  "https://vortex-ci-benchmark-results.s3.amazonaws.com/data.json.gz";
const COMMITS_URL =
  process.env.COMMITS_URL ||
  "https://vortex-ci-benchmark-results.s3.amazonaws.com/commits.json";
const REFRESH_INTERVAL = process.env.REFRESH_INTERVAL || 5 * 60 * 1000;
const USE_LOCAL_DATA = process.env.USE_LOCAL_DATA === "true";
const CACHE_DIR = process.env.CACHE_DIR || undefined;

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

const benchmarks = new BenchmarkStore({
  dataUrl: DATA_URL,
  commitsUrl: COMMITS_URL,
  useLocalData: USE_LOCAL_DATA,
  cacheDir: CACHE_DIR,
});
let refreshIntervalId = null;

const json = (res, code, data) => {
  res.writeHead(code, {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Cache-Control": "no-store",
    Pragma: "no-cache",
  });
  res.end(JSON.stringify(data));
};

function getLoadingPayload() {
  const status = benchmarks.status;
  return {
    error: status.state === "error" ? "Initial refresh failed" : "Loading",
    status: status.state,
    ready: status.ready,
    refreshing: status.refreshing,
    hasData: status.hasData,
    lastUpdated: status.lastUpdated,
    lastRefreshStartedAt: status.lastRefreshStartedAt,
    lastRefreshCompletedAt: status.lastRefreshCompletedAt,
    lastRefreshError: status.lastRefreshError,
  };
}

function triggerRefresh() {
  console.log("Refreshing data...");
  benchmarks.refresh().catch(() => {});
}

function serveFile(res, filePath) {
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(err.code === "ENOENT" ? 404 : 500);
      res.end(err.code === "ENOENT" ? "Not Found" : "Error");
      return;
    }

    const ext = path.extname(filePath).toLowerCase();
    const headers = { "Content-Type": MIME[ext] || "application/octet-stream" };
    if (ext === ".js") {
      headers["Cache-Control"] = "no-cache";
      headers.Pragma = "no-cache";
    }

    res.writeHead(200, headers);
    res.end(data);
  });
}

async function handleData(res, groupName, chartName, params) {
  try {
    const payload = await benchmarks.getChartData(groupName, chartName, {
      start: params.get("start"),
      end: params.get("end"),
      last: params.get("last"),
      startIdx: params.has("startIdx") ? params.get("startIdx") : null,
      endIdx: params.has("endIdx") ? params.get("endIdx") : null,
    });
    json(res, 200, payload);
  } catch (error) {
    if (error.message === "Loading") {
      json(res, 503, getLoadingPayload());
      return;
    }
    if (error.statusCode === 404) {
      json(res, 404, { error: error.message });
      return;
    }
    console.error("Data request error:", error);
    json(res, 500, { error: "Internal server error" });
  }
}

const server = http.createServer((req, res) => {
  const [pathName, rawQuery] = req.url.split("?");
  const params = new URLSearchParams(rawQuery || "");

  if (req.method === "OPTIONS") {
    res.writeHead(204, {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET",
      "Access-Control-Allow-Headers": "Content-Type",
    });
    res.end();
    return;
  }

  if (pathName === "/api/metadata") {
    const metadata = benchmarks.metadata;
    json(res, metadata ? 200 : 503, metadata || getLoadingPayload());
    return;
  }

  if (pathName === "/api/health") {
    const status = benchmarks.status;
    json(res, status.ready ? 200 : 503, status);
    return;
  }

  if (pathName.startsWith("/api/data/")) {
    const segments = pathName.slice(10).split("/");
    handleData(
      res,
      decodeURIComponent(segments[0] || ""),
      decodeURIComponent(segments.slice(1).join("/") || ""),
      params,
    ).catch((error) => {
      console.error("Unhandled data handler error:", error);
      json(res, 500, { error: "Internal server error" });
    });
    return;
  }

  const filePath = path.join(
    __dirname,
    "dist",
    pathName === "/" ? "index.html" : pathName,
  );
  if (!filePath.startsWith(__dirname) || filePath.includes("/sample/")) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }
  serveFile(res, filePath);
});

function start() {
  console.log("Starting server...");
  server.listen(PORT, () => {
    console.log(`Server at http://localhost:${PORT}`);
  });
  triggerRefresh();
  refreshIntervalId = setInterval(triggerRefresh, Number(REFRESH_INTERVAL));
}

async function shutdown() {
  if (refreshIntervalId) {
    clearInterval(refreshIntervalId);
    refreshIntervalId = null;
  }
  await benchmarks.close();
  server.close();
}

process.on("SIGINT", () => {
  shutdown().finally(() => process.exit(0));
});

process.on("SIGTERM", () => {
  shutdown().finally(() => process.exit(0));
});

start();
