// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/**
 * Opaque slugs for `/api/chart/{slug}` and `/api/group/{slug}`, the TypeScript
 * port of `server/src/slug.rs`.
 *
 * The web-ui treats slugs as opaque strings: it receives them from
 * `/api/groups` and feeds them back unchanged, so this read service both
 * produces and consumes them. A slug is `<prefix>.<base64url-of-json>`, where
 * `<prefix>` names the source fact table (from [`./families`]) and the JSON
 * encodes the chart or group key. The JSON keys are emitted in the same order
 * the Rust structs declare them, with the `k` discriminant first (serde's
 * internally-tagged representation), and `Option` fields are emitted as
 * explicit `null` rather than omitted, mirroring serde's default. Round-tripping
 * a slug back gives a strongly-typed [`ChartKey`] or [`GroupKey`].
 */

import { familyForChartKind, familyForGroupKind, type ChartKind, type GroupKind } from './families';

/** Strongly-typed chart key parsed from a slug. Discriminated on `k`. */
export type ChartKey =
  | {
      k: 'QueryMeasurement';
      dataset: string;
      dataset_variant: string | null;
      scale_factor: string | null;
      storage: string;
      query_idx: number;
    }
  | { k: 'CompressionTime'; dataset: string; dataset_variant: string | null }
  | { k: 'CompressionSize'; dataset: string; dataset_variant: string | null }
  | { k: 'RandomAccess'; dataset: string }
  | { k: 'VectorSearch'; dataset: string; layout: string; threshold: number };

/** Strongly-typed group key parsed from a slug. Discriminated on `k`. */
export type GroupKey =
  | {
      k: 'QueryGroup';
      dataset: string;
      dataset_variant: string | null;
      scale_factor: string | null;
      storage: string;
    }
  | { k: 'CompressionTimeGroup' }
  | { k: 'CompressionSizeGroup' }
  | { k: 'RandomAccessGroup' }
  | { k: 'VectorSearchGroup'; dataset: string; layout: string };

const CHART_KINDS: readonly ChartKind[] = [
  'QueryMeasurement',
  'CompressionTime',
  'CompressionSize',
  'RandomAccess',
  'VectorSearch',
];

const GROUP_KINDS: readonly GroupKind[] = [
  'QueryGroup',
  'CompressionTimeGroup',
  'CompressionSizeGroup',
  'RandomAccessGroup',
  'VectorSearchGroup',
];

function encodePayload(json: string): string {
  return Buffer.from(json, 'utf8').toString('base64url');
}

function decodePayload(slug: string): unknown {
  const dot = slug.indexOf('.');
  if (dot === -1) {
    throw new Error("slug missing '.' separator");
  }
  // The prefix before the '.' is ignored on decode (it is a redundant family
  // hint); the JSON payload is the source of truth, matching `slug.rs`.
  const encoded = slug.slice(dot + 1);
  const decoded = Buffer.from(encoded, 'base64url').toString('utf8');
  // An empty decode (e.g. all-invalid base64url) yields '', and `JSON.parse('')`
  // throws, so a malformed payload is rejected here.
  return JSON.parse(decoded) as unknown;
}

/**
 * Build the canonical JSON for a chart key with keys in the Rust declaration
 * order (`k` first), then render `<prefix>.<base64url-of-json>`.
 */
export function chartKeyToSlug(key: ChartKey): string {
  const prefix = familyForChartKind(key.k).chartSlugPrefix;
  let ordered: Record<string, unknown>;
  switch (key.k) {
    case 'QueryMeasurement':
      ordered = {
        k: key.k,
        dataset: key.dataset,
        dataset_variant: key.dataset_variant,
        scale_factor: key.scale_factor,
        storage: key.storage,
        query_idx: key.query_idx,
      };
      break;
    case 'CompressionTime':
    case 'CompressionSize':
      ordered = { k: key.k, dataset: key.dataset, dataset_variant: key.dataset_variant };
      break;
    case 'RandomAccess':
      ordered = { k: key.k, dataset: key.dataset };
      break;
    case 'VectorSearch':
      ordered = { k: key.k, dataset: key.dataset, layout: key.layout, threshold: key.threshold };
      break;
  }
  return `${prefix}.${encodePayload(JSON.stringify(ordered))}`;
}

function asObject(parsed: unknown, what: string): Record<string, unknown> {
  if (typeof parsed !== 'object' || parsed === null) {
    throw new Error(`slug payload is not a ${what} object`);
  }
  return parsed as Record<string, unknown>;
}

function reqString(obj: Record<string, unknown>, field: string): string {
  const value = obj[field];
  if (typeof value !== 'string') {
    throw new Error(`slug payload field \`${field}\` must be a string`);
  }
  return value;
}

// Mirrors serde's `Option<String>`: an absent field deserializes to `None`
// (here `null`), and a present field must be a string or explicit `null`.
function optString(obj: Record<string, unknown>, field: string): string | null {
  const value = obj[field];
  if (value === undefined || value === null) {
    return null;
  }
  if (typeof value !== 'string') {
    throw new Error(`slug payload field \`${field}\` must be a string or null`);
  }
  return value;
}

// Mirrors serde's `f64` deserialization, which rejects an overflowing JSON
// number literal (for example `1e400`) as "number out of range". JavaScript's
// `JSON.parse` instead overflows to `Infinity`, so without the finiteness check
// a forged threshold of `1e400` would survive decode here while the Rust server
// maps the same payload to the malformed-slug 400 path.
function reqNumber(obj: Record<string, unknown>, field: string): number {
  const value = obj[field];
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new Error(`slug payload field \`${field}\` must be a finite number`);
  }
  return value;
}

// Mirrors serde's `i32` deserialization: the value must be a whole number inside
// the signed-32-bit range. The Rust `ChartKey::QueryMeasurement.query_idx` is an
// `i32`, so `from_slug` rejects a non-integer (`1.5`) or out-of-range (`2**31`)
// payload as a malformed slug. Without this check a forged `query_idx` survives
// decode and only fails later at the Postgres `integer` bind, turning a 400 into
// an unhandled 500.
const I32_MIN = -2_147_483_648;
const I32_MAX = 2_147_483_647;
function reqI32(obj: Record<string, unknown>, field: string): number {
  const value = reqNumber(obj, field);
  if (!Number.isInteger(value) || value < I32_MIN || value > I32_MAX) {
    throw new Error(`slug payload field \`${field}\` must be a 32-bit integer`);
  }
  return value;
}

/**
 * Parse a slug previously produced by [`chartKeyToSlug`]. Validates the full
 * payload shape per variant (not just the `k` discriminant), so a
 * known-discriminant-but-malformed payload is rejected rather than returned as
 * a `ChartKey` with `undefined` required fields. This matches serde's
 * reject-on-bad-shape contract and keeps the cast sound by constructing the
 * result from validated fields.
 */
export function chartKeyFromSlug(slug: string): ChartKey {
  const obj = asObject(decodePayload(slug), 'chart key');
  const k = obj.k;
  if (typeof k !== 'string' || !(CHART_KINDS as readonly string[]).includes(k)) {
    throw new Error(`slug payload has unknown chart key discriminant \`${String(k)}\``);
  }
  const kind = k as ChartKind;
  switch (kind) {
    case 'QueryMeasurement':
      return {
        k: kind,
        dataset: reqString(obj, 'dataset'),
        dataset_variant: optString(obj, 'dataset_variant'),
        scale_factor: optString(obj, 'scale_factor'),
        storage: reqString(obj, 'storage'),
        query_idx: reqI32(obj, 'query_idx'),
      };
    case 'CompressionTime':
    case 'CompressionSize':
      return {
        k: kind,
        dataset: reqString(obj, 'dataset'),
        dataset_variant: optString(obj, 'dataset_variant'),
      };
    case 'RandomAccess':
      return { k: kind, dataset: reqString(obj, 'dataset') };
    case 'VectorSearch':
      return {
        k: kind,
        dataset: reqString(obj, 'dataset'),
        layout: reqString(obj, 'layout'),
        threshold: reqNumber(obj, 'threshold'),
      };
  }
}

/**
 * Build the canonical JSON for a group key with keys in the Rust declaration
 * order (`k` first), then render `<prefix>.<base64url-of-json>`.
 */
export function groupKeyToSlug(key: GroupKey): string {
  const prefix = familyForGroupKind(key.k).groupSlugPrefix;
  let ordered: Record<string, unknown>;
  switch (key.k) {
    case 'QueryGroup':
      ordered = {
        k: key.k,
        dataset: key.dataset,
        dataset_variant: key.dataset_variant,
        scale_factor: key.scale_factor,
        storage: key.storage,
      };
      break;
    case 'VectorSearchGroup':
      ordered = { k: key.k, dataset: key.dataset, layout: key.layout };
      break;
    case 'CompressionTimeGroup':
    case 'CompressionSizeGroup':
    case 'RandomAccessGroup':
      ordered = { k: key.k };
      break;
  }
  return `${prefix}.${encodePayload(JSON.stringify(ordered))}`;
}

/**
 * Parse a slug previously produced by [`groupKeyToSlug`]. Validates the full
 * payload shape per variant (see [`chartKeyFromSlug`] for the rationale).
 */
export function groupKeyFromSlug(slug: string): GroupKey {
  const obj = asObject(decodePayload(slug), 'group key');
  const k = obj.k;
  if (typeof k !== 'string' || !(GROUP_KINDS as readonly string[]).includes(k)) {
    throw new Error(`slug payload has unknown group key discriminant \`${String(k)}\``);
  }
  const kind = k as GroupKind;
  switch (kind) {
    case 'QueryGroup':
      return {
        k: kind,
        dataset: reqString(obj, 'dataset'),
        dataset_variant: optString(obj, 'dataset_variant'),
        scale_factor: optString(obj, 'scale_factor'),
        storage: reqString(obj, 'storage'),
      };
    case 'VectorSearchGroup':
      return { k: kind, dataset: reqString(obj, 'dataset'), layout: reqString(obj, 'layout') };
    case 'CompressionTimeGroup':
    case 'CompressionSizeGroup':
    case 'RandomAccessGroup':
      return { k: kind };
  }
}
