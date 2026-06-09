// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/**
 * The read service's `SCHEMA_VERSION` lockstep site (plan Table D).
 *
 * This constant must stay equal to the other lockstep sites in one PR or CI
 * ingest 400/409s:
 *
 * - `benchmarks-website/server/src/schema.rs` (`pub const SCHEMA_VERSION: i32`).
 * - `vortex-bench/src/v3.rs`.
 * - `scripts/post-ingest.py` (`SCHEMA_VERSION`).
 * - `benchmarks-website/migrate/src/lib.rs` (`pub const SCHEMA_VERSION`).
 *
 * The read service surfaces it on `/health` so an operator can detect envelope
 * or schema skew between the served data and the producers.
 */
export const SCHEMA_VERSION = 1;
