import { FAN_OUT_GROUPS, QUERY_SUITES } from "../src/config.js";

export const MAX_POINTS = 200;
export const DUCKDB_OPTIONS = { threads: "4" };
export const DEFAULT_CACHE_DIR_NAME = "vortex-benchmarks-website-cache";

export const GROUPS = [
  "Random Access",
  "Compression",
  "Compression Size",
  ...QUERY_SUITES.filter((suite) => !suite.skip && !suite.fanOut).map(
    (suite) => suite.displayName,
  ),
  ...FAN_OUT_GROUPS,
];

export const QUERY_GROUP_EXCLUSIONS = new Set([
  "Random Access",
  "Compression",
  "Compression Size",
]);
