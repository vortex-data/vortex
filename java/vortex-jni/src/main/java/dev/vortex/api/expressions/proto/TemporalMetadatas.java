/**
 * (c) Copyright 2025 SpiralDB Inc. All rights reserved.
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package dev.vortex.api.expressions.proto;

import com.google.common.base.Preconditions;
import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Optional;

final class TemporalMetadatas {
    private TemporalMetadatas() {}

    static byte TIME_UNIT_NANOS = 0;
    static byte TIME_UNIT_MICROS = 1;
    static byte TIME_UNIT_MILLIS = 2;
    static byte TIME_UNIT_SECONDS = 3;
    static byte TIME_UNIT_DAYS = 4;

    static final Supplier<byte[]> DATE_DAYS = Suppliers.memoize(() -> new byte[] {TIME_UNIT_DAYS});
    static final Supplier<byte[]> DATE_MILLIS = Suppliers.memoize(() -> new byte[] {TIME_UNIT_MILLIS});
    static final Supplier<byte[]> TIME_SECONDS = Suppliers.memoize(() -> new byte[] {TIME_UNIT_SECONDS});
    static final Supplier<byte[]> TIME_MILLIS = Suppliers.memoize(() -> new byte[] {TIME_UNIT_MILLIS});
    static final Supplier<byte[]> TIME_MICROS = Suppliers.memoize(() -> new byte[] {TIME_UNIT_MICROS});
    static final Supplier<byte[]> TIME_NANOS = Suppliers.memoize(() -> new byte[] {TIME_UNIT_NANOS});

    static byte[] timestamp(byte timeUnit, Optional<String> timeZone) {
        Preconditions.checkArgument(
                timeUnit >= TIME_UNIT_NANOS && timeUnit < TIME_UNIT_DAYS, "invalid timeUnit for Timestamp:" + timeUnit);

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        baos.write(timeUnit);
        if (timeZone.isPresent()) {
            byte[] timeZoneBytes = timeZone.get().getBytes(StandardCharsets.UTF_8);
            // Write length as little-endian uint16.
            int lenLow = timeZoneBytes.length & 0xFF;
            int lenHigh = (timeZoneBytes.length >> 8) & 0xFF;
            baos.write(lenLow);
            baos.write(lenHigh);
            baos.writeBytes(timeZoneBytes);
        } else {
            // write uint16 zero value
            baos.write(0);
            baos.write(0);
        }
        return baos.toByteArray();
    }

    /**
     * Extract the time unit byte from the serialized metadata.
     */
    static byte getTimeUnit(byte[] serializedMetadata) {
        byte timeUnit = serializedMetadata[0];
        Preconditions.checkArgument(
                timeUnit >= TIME_UNIT_NANOS && timeUnit <= TIME_UNIT_DAYS, "invalid timeUnit byte: " + timeUnit);

        return timeUnit;
    }

    /**
     * Read from a serialized metadata representation into a time zone string.
     */
    static Optional<String> getTimeZone(byte[] serializedMetadata) {
        byte _timeUnit = serializedMetadata[0];
        byte lenLow = serializedMetadata[1];
        byte lenHigh = serializedMetadata[2];
        int len = ((lenHigh & 0xFF) << 8) | (lenLow & 0xFF);
        if (len == 0) {
            return Optional.empty();
        } else {
            byte[] timeZoneBytes = new byte[len];
            System.arraycopy(serializedMetadata, 3, timeZoneBytes, 0, len);
            return Optional.of(new String(timeZoneBytes, StandardCharsets.UTF_8));
        }
    }
}
