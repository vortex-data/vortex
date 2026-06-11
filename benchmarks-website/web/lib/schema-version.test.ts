// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import { describe, expect, it } from 'vitest';
import { SCHEMA_VERSION } from './schema-version';

describe('SCHEMA_VERSION', () => {
  // Drift sentinel for the cross-language lockstep (plan Table D / BANS): this
  // MUST stay equal to `server/src/schema.rs`'s `SCHEMA_VERSION` (= 1). A direct
  // assertion fails loud the moment the TS const is edited out of lockstep,
  // rather than only surfacing transitively through the /health assembly tests.
  it('is pinned to 1, matching server/src/schema.rs', () => {
    expect(SCHEMA_VERSION).toBe(1);
  });
});
