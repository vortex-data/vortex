// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api.expressions.proto;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.vortex.jni.NativeFileMethods;
import org.junit.jupiter.api.Test;

public final class TestOpenErrors {
    @Test
    public void testOpenThrows() {
        RuntimeException runtimeException = assertThrows(RuntimeException.class, () -> {
            NativeFileMethods.open("bad_scheme:///fake-location", null);
        });
        assertTrue(runtimeException.getMessage().contains("Invalid URL"));
    }
}
