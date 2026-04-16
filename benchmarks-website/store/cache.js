import fs from "fs";
import fsp from "fs/promises";
import os from "os";
import path from "path";
import { Readable } from "stream";
import { pipeline } from "stream/promises";
import { fileURLToPath } from "url";
import { DEFAULT_CACHE_DIR_NAME } from "./constants.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const WEBSITE_DIR = path.dirname(__dirname);

function cachePaths(cacheDir) {
  return {
    dataPath: path.join(cacheDir, "data.json.gz"),
    commitsPath: path.join(cacheDir, "commits.json"),
    manifestPath: path.join(cacheDir, "manifest.json"),
  };
}

async function pathExists(filePath) {
  try {
    await fsp.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function readJsonFile(filePath) {
  try {
    const raw = await fsp.readFile(filePath, "utf8");
    return JSON.parse(raw);
  } catch (error) {
    if (error.code === "ENOENT") return null;
    throw error;
  }
}

async function writeJsonFileAtomic(filePath, value) {
  const tempPath = `${filePath}.tmp-${process.pid}-${Date.now()}`;
  await fsp.writeFile(tempPath, JSON.stringify(value, null, 2));
  await fsp.rename(tempPath, filePath);
}

async function downloadToFile(url, destination, metadata = {}) {
  const headers = {};
  if (metadata.etag) headers["If-None-Match"] = metadata.etag;
  if (metadata.lastModified) headers["If-Modified-Since"] = metadata.lastModified;

  const response = await fetch(url, { headers });
  if (response.status === 304) {
    return {
      changed: false,
      metadata: {
        ...metadata,
        checkedAt: new Date().toISOString(),
      },
    };
  }

  if (!response.ok) {
    throw new Error(`Fetch failed: ${url} ${response.status}`);
  }

  if (!response.body) {
    throw new Error(`Fetch failed: ${url} empty body`);
  }

  const tempPath = `${destination}.tmp-${process.pid}-${Date.now()}`;
  try {
    await pipeline(Readable.fromWeb(response.body), fs.createWriteStream(tempPath));
    await fsp.rename(tempPath, destination);
  } finally {
    if (await pathExists(tempPath)) {
      await fsp.unlink(tempPath);
    }
  }

  return {
    changed: true,
    metadata: {
      url,
      etag: response.headers.get("etag"),
      lastModified: response.headers.get("last-modified"),
      fetchedAt: new Date().toISOString(),
      checkedAt: new Date().toISOString(),
    },
  };
}

export async function prepareInputFiles({
  dataUrl,
  commitsUrl,
  useLocalData,
  cacheDir,
  preferCached = false,
  forceRemoteCheck = false,
}) {
  if (useLocalData) {
    return {
      dataPath: path.join(WEBSITE_DIR, "sample/data.json"),
      commitsPath: path.join(WEBSITE_DIR, "sample/commits.json"),
      changed: true,
      source: "local",
      deferRemoteCheck: false,
    };
  }

  const resolvedCacheDir =
    cacheDir || path.join(os.tmpdir(), DEFAULT_CACHE_DIR_NAME);
  await fsp.mkdir(resolvedCacheDir, { recursive: true });

  const { dataPath, commitsPath, manifestPath } = cachePaths(resolvedCacheDir);
  const manifest = (await readJsonFile(manifestPath)) || {};
  const [hasDataFile, hasCommitsFile] = await Promise.all([
    pathExists(dataPath),
    pathExists(commitsPath),
  ]);
  const hasCachedFiles = hasDataFile && hasCommitsFile;

  if (preferCached && hasCachedFiles && !forceRemoteCheck) {
    return {
      dataPath,
      commitsPath,
      changed: true,
      source: "cache",
      deferRemoteCheck: true,
    };
  }

  try {
    const [dataResult, commitsResult] = await Promise.all([
      downloadToFile(dataUrl, dataPath, manifest.data || {}),
      downloadToFile(commitsUrl, commitsPath, manifest.commits || {}),
    ]);

    await writeJsonFileAtomic(manifestPath, {
      version: 1,
      dataUrl,
      commitsUrl,
      data: dataResult.metadata,
      commits: commitsResult.metadata,
      updatedAt: new Date().toISOString(),
    });

    return {
      dataPath,
      commitsPath,
      changed: !hasCachedFiles || dataResult.changed || commitsResult.changed,
      source:
        !hasCachedFiles || dataResult.changed || commitsResult.changed
          ? "remote"
          : "cache",
      deferRemoteCheck: false,
    };
  } catch (error) {
    if (hasCachedFiles) {
      console.warn(
        `Falling back to cached benchmark files after refresh failed: ${error.message}`,
      );
      return {
        dataPath,
        commitsPath,
        changed: false,
        source: "stale-cache",
        deferRemoteCheck: false,
      };
    }

    throw error;
  }
}
