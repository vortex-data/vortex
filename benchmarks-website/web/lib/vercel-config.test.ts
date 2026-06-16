// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

// Read vercel.json from disk (not an import) so the assertion pins the actual
// shipped config the way Vercel reads it.
const vercelConfig = JSON.parse(
  readFileSync(fileURLToPath(new URL('../vercel.json', import.meta.url)), 'utf8'),
) as { crons?: Array<{ path: string; schedule: string }> };

describe('vercel.json keep-warm cron', () => {
  it('pings /api/health every 2 minutes to keep the function + DB pool warm', () => {
    expect(vercelConfig.crons).toContainEqual({
      path: '/api/health',
      schedule: '*/2 * * * *',
    });
  });
});
