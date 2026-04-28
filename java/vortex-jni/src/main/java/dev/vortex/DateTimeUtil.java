// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.OffsetDateTime;
import java.time.ZoneOffset;
import java.time.temporal.ChronoUnit;

/**
 * Utility class for date and time conversions in Vortex.
 *
 * <p>This class provides helper functions for converting Java date/time objects to nanoseconds since the Unix epoch,
 * which is the internal representation used by Vortex for temporal data.
 *
 * <p>Helpful functions borrowed from Iceberg class with same name.
 */
public final class DateTimeUtil {
    /** The Unix epoch as an OffsetDateTime (1970-01-01T00:00:00Z). */
    public static final OffsetDateTime EPOCH = Instant.ofEpochSecond(0).atOffset(ZoneOffset.UTC);

    /**
     * Converts a LocalDateTime to nanoseconds since the Unix epoch.
     *
     * <p>The LocalDateTime is assumed to be in UTC timezone for conversion purposes. When decoded back to
     * LocalDateTime, the timezone information will not be preserved.
     *
     * @param dateTime the LocalDateTime to convert
     * @return nanoseconds since the Unix epoch
     */
    public static long nanosFromTimestamp(LocalDateTime dateTime) {
        return ChronoUnit.NANOS.between(EPOCH, dateTime.atOffset(ZoneOffset.UTC));
    }

    /**
     * Converts an OffsetDateTime to nanoseconds since the Unix epoch.
     *
     * <p>The timezone information in the OffsetDateTime is preserved during conversion.
     *
     * @param dateTime the OffsetDateTime to convert
     * @return nanoseconds since the Unix epoch
     */
    public static long nanosFromTimestamptz(OffsetDateTime dateTime) {
        return ChronoUnit.NANOS.between(EPOCH, dateTime);
    }
}
