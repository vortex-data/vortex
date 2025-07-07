// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api.expressions.proto;

import com.google.protobuf.ByteString;
import java.math.BigDecimal;
import java.math.BigInteger;

final class EndianUtils {
    static byte[] reverse(ByteString src) {
        byte[] dst = new byte[src.size()];
        for (int i = 0; i < dst.length; i++) {
            dst[i] = src.byteAt(dst.length - 1 - i);
        }
        return dst;
    }

    static byte[] littleEndianDecimal(BigDecimal decimal) {
        BigInteger unscaled = decimal.unscaledValue();
        byte[] bigEndianBytes = unscaled.toByteArray();

        // Determine target size (1, 2, 4, 8, 16, or 32 bytes)
        int targetSize;
        if (bigEndianBytes.length <= 1) {
            targetSize = 1;
        } else if (bigEndianBytes.length <= 2) {
            targetSize = 2;
        } else if (bigEndianBytes.length <= 4) {
            targetSize = 4;
        } else if (bigEndianBytes.length <= 8) {
            targetSize = 8;
        } else if (bigEndianBytes.length <= 16) {
            targetSize = 16;
        } else if (bigEndianBytes.length <= 32) {
            targetSize = 32;
        } else {
            throw new IllegalArgumentException(
                    "BigDecimal with " + bigEndianBytes.length + " bytes overflows maximum Vortex decimal size");
        }

        byte[] result = new byte[targetSize];

        // Copy bytes in reverse order (big endian to little endian)
        for (int i = 0; i < bigEndianBytes.length; i++) {
            result[i] = bigEndianBytes[bigEndianBytes.length - 1 - i];
        }

        // Sign extend if negative
        if (unscaled.signum() < 0) {
            for (int i = bigEndianBytes.length; i < targetSize; i++) {
                result[i] = (byte) 0xFF;
            }
        }

        return result;
    }
}
