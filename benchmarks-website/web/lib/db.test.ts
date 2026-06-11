// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import type { StartedPostgreSqlContainer } from '@testcontainers/postgresql';
import { Signer } from '@aws-sdk/rds-signer';
import { dockerAvailable, startBenchContainer } from './test-harness';
import {
  buildQuery,
  passwordProvider,
  requireEnv,
  resetPool,
  resolveSsl,
  sql,
  type DbConfig,
} from './db';

// A single shared getAuthToken mock so the IAM test can script distinct
// per-call return values and assert the call count. Hoisted so it is defined
// before the (hoisted) vi.mock factory references it.
const { getAuthTokenMock } = vi.hoisted(() => ({ getAuthTokenMock: vi.fn() }));

// Mock the RDS signer so the IAM token path is exercised without a live AWS
// endpoint. The testcontainers roundtrip below uses a static password, so it
// never constructs a Signer and is unaffected by this mock.
vi.mock('@aws-sdk/rds-signer', () => ({
  // A `function` (not an arrow) so the mock is constructable via `new Signer()`.
  Signer: vi.fn(function MockSigner() {
    return { getAuthToken: getAuthTokenMock };
  }),
}));

describe.skipIf(!dockerAvailable())('db pool roundtrip (testcontainers Postgres)', () => {
  let container: StartedPostgreSqlContainer;

  beforeAll(async () => {
    // The pool roundtrip needs no schema; the BENCH_DB_PASSWORD fixture path
    // set by the harness means IAM token generation is bypassed.
    container = await startBenchContainer({ applySchema: false });
  });

  afterAll(async () => {
    await resetPool();
    await container.stop();
  });

  it('connects via the password fixture and roundtrips a SELECT', async () => {
    const rows = await sql<{
      one: number;
      greeting: string;
    }>`SELECT ${1}::int AS one, ${'hi'}::text AS greeting`;
    expect(rows).toEqual([{ one: 1, greeting: 'hi' }]);
  });

  it('binds interpolated values as parameters rather than concatenating them', async () => {
    const hostile = '1); DROP TABLE x; --';
    const rows = await sql<{ v: string }>`SELECT ${hostile}::text AS v`;
    // The hostile string round-trips verbatim as a value, proving it was bound
    // as $1 and never interpolated into the SQL text.
    expect(rows).toEqual([{ v: hostile }]);
  });
});

describe('db IAM auth path (mocked rds-signer)', () => {
  const iamConfig: DbConfig = {
    host: 'proxy.example.us-east-1.rds.amazonaws.com',
    port: 5432,
    database: 'bench',
    user: 'bench_reader',
    region: 'us-east-1',
    ssl: false,
    poolMax: 4,
    staticPassword: undefined,
  };

  it('mints a FRESH RDS IAM token per connection (not a single cached token)', async () => {
    vi.mocked(Signer).mockClear();
    getAuthTokenMock.mockReset();
    getAuthTokenMock.mockResolvedValueOnce('iam-token-1').mockResolvedValueOnce('iam-token-2');

    const provider = passwordProvider(iamConfig);
    // pg invokes the provider once per new physical connection; distinct tokens
    // across two calls prove a fresh mint each time rather than one cached token.
    await expect(provider()).resolves.toBe('iam-token-1');
    await expect(provider()).resolves.toBe('iam-token-2');
    expect(getAuthTokenMock).toHaveBeenCalledTimes(2);
    expect(Signer).toHaveBeenCalledWith({
      hostname: 'proxy.example.us-east-1.rds.amazonaws.com',
      port: 5432,
      username: 'bench_reader',
      region: 'us-east-1',
    });
  });

  it('uses the static password and skips IAM when one is supplied', async () => {
    const provider = passwordProvider({ ...iamConfig, region: '', staticPassword: 'fixture-pw' });
    await expect(provider()).resolves.toBe('fixture-pw');
  });

  it('throws when neither a static password nor a region is configured', () => {
    expect(() => passwordProvider({ ...iamConfig, region: '' })).toThrow(/BENCH_DB_REGION/);
  });
});

describe('buildQuery (parameterization)', () => {
  const q = (strings: TemplateStringsArray, ...values: unknown[]) => buildQuery(strings, values);

  it('numbers interpolated values $1..$n positionally', () => {
    expect(q`SELECT ${1}, ${'x'}, ${true}`).toEqual({
      text: 'SELECT $1, $2, $3',
      values: [1, 'x', true],
    });
  });

  it('emits no placeholders for a template with no interpolations', () => {
    expect(q`SELECT 1`).toEqual({ text: 'SELECT 1', values: [] });
  });

  it('handles leading and trailing interpolation', () => {
    expect(q`${'a'} mid ${42}`).toEqual({ text: '$1 mid $2', values: ['a', 42] });
  });
});

describe('resolveSsl', () => {
  afterEach(() => {
    delete process.env.BENCH_DB_SSL;
    delete process.env.BENCH_DB_CA;
  });

  it('defaults to verify-full (rejectUnauthorized true)', () => {
    delete process.env.BENCH_DB_SSL;
    expect(resolveSsl()).toEqual({ rejectUnauthorized: true });
  });

  it('returns false for mode=disable', () => {
    process.env.BENCH_DB_SSL = 'disable';
    expect(resolveSsl()).toBe(false);
  });

  it('merges BENCH_DB_CA into the verify-full ssl object', () => {
    process.env.BENCH_DB_SSL = 'verify-full';
    process.env.BENCH_DB_CA = 'rds-ca-pem';
    expect(resolveSsl()).toEqual({ rejectUnauthorized: true, ca: 'rds-ca-pem' });
  });

  it('throws (fails loud) on an unrecognized mode rather than silently disabling verification', () => {
    process.env.BENCH_DB_SSL = 'verify-ca';
    expect(() => resolveSsl()).toThrow(/BENCH_DB_SSL/);
  });
});

describe('requireEnv', () => {
  it('returns a set value', () => {
    process.env.BENCH_TEST_REQ = 'present';
    expect(requireEnv('BENCH_TEST_REQ')).toBe('present');
    delete process.env.BENCH_TEST_REQ;
  });

  it('throws on a missing variable', () => {
    delete process.env.BENCH_TEST_REQ_MISSING;
    expect(() => requireEnv('BENCH_TEST_REQ_MISSING')).toThrow(/BENCH_TEST_REQ_MISSING/);
  });

  it('throws on an empty variable', () => {
    process.env.BENCH_TEST_REQ_EMPTY = '';
    expect(() => requireEnv('BENCH_TEST_REQ_EMPTY')).toThrow();
    delete process.env.BENCH_TEST_REQ_EMPTY;
  });
});
