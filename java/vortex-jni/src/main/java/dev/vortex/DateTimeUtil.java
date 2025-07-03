// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.time.temporal.ChronoUnit;

// Helpful functions borrowed from Iceberg class with same name.
public final class DateTimeUtil {
    public static final OffsetDateTime EPOCH = Instant.ofEpochSecond(0).atOffset(ZoneOffset.UTC);

    // Just assume timestamp is already at UTC for conversion purposes.
    // When decoded back to LocalDateTime
    public static long nanosFromTimestamp(LocalDateTime dateTime) {
        return ChronoUnit.NANOS.between(EPOCH, dateTime.atOffset(ZoneOffset.UTC));
    }

    public static long nanosFromTimestamptz(OffsetDateTime dateTime) {
        return ChronoUnit.NANOS.between(EPOCH, dateTime);
    }
}
