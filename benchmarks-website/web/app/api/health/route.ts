// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { NextResponse } from 'next/server';

import { collectHealth } from '@/lib/health';

// A liveness probe must reflect the live database, never a cached snapshot.
export const dynamic = 'force-dynamic';

/**
 * `GET /api/health` returns the v3-compatible snake_case `HealthResponse`
 * (status, per-table `row_counts`, DB host, build SHA, `schema_version`); see
 * [`collectHealth`] for the wire shape.
 */
export async function GET() {
  return NextResponse.json(await collectHealth());
}
