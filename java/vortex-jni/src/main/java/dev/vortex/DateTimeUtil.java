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
