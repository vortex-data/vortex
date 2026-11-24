// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex;

import java.lang.ref.Cleaner;

public final class VortexCleaner {
    private static final Cleaner cleaner = Cleaner.create();

    public static Cleaner.Cleanable register(Object obj, Runnable r) {
        return VortexCleaner.cleaner.register(obj, r);
    }
}
