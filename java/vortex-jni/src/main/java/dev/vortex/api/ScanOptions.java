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
package dev.vortex.api;

import java.util.List;
import java.util.Optional;
import org.immutables.value.Value;

/**
 * Create a new set of options for configuring the scan.
 */
@Value.Immutable
public interface ScanOptions {
    /**
     * Columns to project out.
     */
    List<String> columns();

    /**
     * Optional pruning expression that is pushed down to the scan.
     */
    Optional<Expression> predicate();

    /**
     * Optional row indices to select specific rows.
     * These must be sorted in ascending order.
     */
    Optional<long[]> rowIndices();

    static ScanOptions of() {
        return ImmutableScanOptions.builder().build();
    }

    static ImmutableScanOptions.Builder builder() {
        return ImmutableScanOptions.builder();
    }
}
