// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { Pool, type PoolConfig, type QueryResultRow } from 'pg';
import { Signer } from '@aws-sdk/rds-signer';

/**
 * Resolved Postgres connection settings for the benchmarks read service.
 *
 * In production the read service runs on Vercel; the endpoint and auth choice
 * (the public RDS instance endpoint unless the Vercel project gains VPC
 * connectivity to the VPC-internal RDS Proxy, IAM tokens or a static reader
 * password) is configured per environment, see the README's "Database
 * environment" and "Deploys" sections. On the IAM path, RDS IAM auth tokens
 * are valid for only ~15 minutes, so a fresh token is minted for every new
 * pool connection (see [`passwordProvider`]); that is the
 * token-refresh-before-expiry strategy.
 *
 * When [`DbConfig.staticPassword`] is set (local development and the
 * integration tests) the IAM token path is skipped entirely and the static
 * password is used, which keeps the test fixture free of any AWS dependency.
 */
export interface DbConfig {
  host: string;
  port: number;
  database: string;
  user: string;
  /** AWS region for IAM token generation; unused when `staticPassword` is set. */
  region: string;
  ssl: PoolConfig['ssl'];
  poolMax: number;
  /** Idle-connection timeout (ms) for the pg pool; see `resolveIdleTimeoutMillis`. */
  idleTimeoutMillis: number;
  /** When defined, IAM token generation is bypassed in favor of this password. */
  staticPassword: string | undefined;
}

/**
 * Read a required environment variable, treating an empty string as missing.
 * Throws rather than defaulting so a misconfigured deployment fails loudly at
 * first connection instead of silently targeting the wrong database.
 */
export function requireEnv(name: string): string {
  const value = process.env[name];
  if (value === undefined || value === '') {
    throw new Error(
      `Missing required environment variable \`${name}\` for the benchmarks DB connection.`,
    );
  }
  return value;
}

/**
 * Resolves the pg SSL option from `BENCH_DB_SSL`. `verify-full` (the default)
 * validates the server certificate chain AND hostname (`rejectUnauthorized:
 * true`). Node validates against its bundled trust store, which does NOT
 * include the Amazon RDS roots, so `BENCH_DB_CA` must be set to the RDS CA
 * bundle in production or the connection fails loudly at connect time.
 * `disable` turns TLS verification off and is used only by the local
 * integration tests against a non-TLS container. Any other value is rejected
 * rather than silently downgrading certificate verification.
 *
 * Exported for unit testing the mode handling.
 */
export function resolveSsl(): PoolConfig['ssl'] {
  const mode = process.env.BENCH_DB_SSL ?? 'verify-full';
  if (mode === 'disable') {
    return false;
  }
  if (mode !== 'verify-full') {
    throw new Error(
      `Unsupported \`BENCH_DB_SSL\` mode \`${mode}\`; expected 'verify-full' (default) or 'disable'.`,
    );
  }
  const ca = process.env.BENCH_DB_CA;
  return { rejectUnauthorized: true, ...(ca ? { ca } : {}) };
}

/** Default pg pool idle-connection timeout: 5 minutes, see `resolveIdleTimeoutMillis`. */
const DEFAULT_IDLE_TIMEOUT_MS = 300_000;

/**
 * Resolves the pool's idle-connection timeout in milliseconds from
 * `BENCH_DB_IDLE_TIMEOUT_MS`. An unset OR empty/whitespace-only value uses the
 * default `DEFAULT_IDLE_TIMEOUT_MS` (5 minutes) so a pooled connection survives
 * the keep-warm cron's two-minute ping gap instead of pg's 10s default, which
 * would otherwise drop the connection between pings and make the next request
 * re-pay the RDS IAM-token + TLS connect even on a warm function instance. `0`
 * is accepted and means pg never times out an idle client. A non-empty,
 * non-numeric, or negative value fails loudly rather than silently becoming
 * `NaN`. Exported for unit testing the parsing and default.
 */
export function resolveIdleTimeoutMillis(): number {
  const raw = process.env.BENCH_DB_IDLE_TIMEOUT_MS;
  // Treat unset OR empty/whitespace-only as "use the default". This is an
  // optional tuning knob, so an accidentally-cleared value falls back to the
  // safe default rather than silently becoming `Number('')` === 0 (no timeout).
  if (raw === undefined || raw.trim() === '') {
    return DEFAULT_IDLE_TIMEOUT_MS;
  }
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(
      `Invalid \`BENCH_DB_IDLE_TIMEOUT_MS\` \`${raw}\`; expected a non-negative number of milliseconds.`,
    );
  }
  return value;
}

function readConfig(): DbConfig {
  const staticPassword = process.env.BENCH_DB_PASSWORD;
  return {
    host: requireEnv('BENCH_DB_HOST'),
    port: Number(process.env.BENCH_DB_PORT ?? '5432'),
    database: requireEnv('BENCH_DB_NAME'),
    user: requireEnv('BENCH_DB_USER'),
    region: process.env.BENCH_DB_REGION ?? '',
    ssl: resolveSsl(),
    poolMax: Number(process.env.BENCH_DB_POOL_MAX ?? '8'),
    idleTimeoutMillis: resolveIdleTimeoutMillis(),
    staticPassword: staticPassword === '' ? undefined : staticPassword,
  };
}

/**
 * Builds the per-connection password resolver passed to [`pg.Pool`]. `pg`
 * invokes the resolver for every new physical connection, so returning a
 * freshly-minted RDS IAM token here means each connection authenticates with a
 * token comfortably inside its ~15-minute validity window.
 *
 * Exported for unit-testing the IAM token path without a live AWS endpoint.
 */
export function passwordProvider(config: DbConfig): () => Promise<string> {
  if (config.staticPassword !== undefined) {
    const password = config.staticPassword;
    return () => Promise.resolve(password);
  }
  if (config.region === '') {
    throw new Error(
      '`BENCH_DB_REGION` is required when no `BENCH_DB_PASSWORD` is set (IAM auth path).',
    );
  }
  const signer = new Signer({
    hostname: config.host,
    port: config.port,
    username: config.user,
    region: config.region,
  });
  return () => signer.getAuthToken();
}

function createPool(config: DbConfig = readConfig()): Pool {
  return new Pool({
    host: config.host,
    port: config.port,
    database: config.database,
    user: config.user,
    password: passwordProvider(config),
    ssl: config.ssl,
    max: config.poolMax,
    idleTimeoutMillis: config.idleTimeoutMillis,
  });
}

// A single pool is shared across warm serverless invocations and survives
// Next.js dev HMR module reloads by caching on `globalThis`.
const globalForPool = globalThis as unknown as { __benchDbPool?: Pool };

/** Returns the process-wide singleton connection pool, creating it on first use. */
export function getPool(): Pool {
  globalForPool.__benchDbPool ??= createPool();
  return globalForPool.__benchDbPool;
}

/**
 * Closes the singleton pool (if any) and clears the cached reference so the
 * next [`getPool`] rebuilds a live pool. Used by tests to release the pool in
 * teardown; in production the pool lives for the process so this is rarely
 * called.
 */
export async function resetPool(): Promise<void> {
  const pool = globalForPool.__benchDbPool;
  if (pool) {
    globalForPool.__benchDbPool = undefined;
    await pool.end();
  }
}

/**
 * Builds the `(text, values)` pair for a parameterized query from a tagged
 * template. Interpolated values become `$1`, `$2`, … bind parameters and are
 * returned positionally; nothing is string-concatenated into the SQL text, so
 * the result is safe against SQL injection by construction. Pure and exported
 * for unit testing the placeholder numbering without a database.
 */
export function buildQuery(
  strings: TemplateStringsArray,
  values: unknown[],
): { text: string; values: unknown[] } {
  const text = strings.reduce(
    (acc, part, i) => acc + part + (i < values.length ? `$${i + 1}` : ''),
    '',
  );
  return { text, values };
}

/**
 * Tagged-template helper for parameterized queries against the pool.
 *
 * ```ts
 * const rows = await sql<{ n: number }>`SELECT ${1}::int AS n`;
 * ```
 *
 * Division of labor: use `sql` for queries whose text is fixed at the call
 * site (for example `lib/health.ts`); SQL assembled dynamically from filters
 * (the chart and summary queries) uses `QueryParams` in `lib/queries.ts`,
 * which numbers placeholders as conditions are pushed.
 */
export async function sql<T extends QueryResultRow = QueryResultRow>(
  strings: TemplateStringsArray,
  ...values: unknown[]
): Promise<T[]> {
  const { text } = buildQuery(strings, values);
  const result = await getPool().query<T>(text, values);
  return result.rows;
}
