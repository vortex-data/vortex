/*
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 */

package dev.vortex;

import java.lang.ref.Cleaner;

import static java.lang.ref.Cleaner.create;

public final class VortexCleaner {
    private static final Cleaner cleaner = create();
}
