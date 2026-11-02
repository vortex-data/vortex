// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

final class FileAggregateTest {
    /** The tags must match {@code AggregateKind::from_tag} in {@code vortex-jni/src/aggregate.rs}. */
    @Test
    void kindTagsMatchTheNativeTable() {
        assertEquals(0, FileAggregate.Kind.COUNT_STAR.tag());
        assertEquals(1, FileAggregate.Kind.COUNT.tag());
        assertEquals(2, FileAggregate.Kind.MIN.tag());
        assertEquals(3, FileAggregate.Kind.MAX.tag());
        assertEquals(4, FileAggregate.Kind.SUM.tag());
        assertEquals(5, FileAggregate.Kind.IS_SORTED.tag());
    }

    @Test
    void onlyCountStarTakesNoColumn() {
        assertThrows(IllegalArgumentException.class, () -> new FileAggregate(FileAggregate.Kind.MIN, null));
        assertThrows(IllegalArgumentException.class, () -> new FileAggregate(FileAggregate.Kind.COUNT_STAR, "id"));
        assertEquals(new FileAggregate(FileAggregate.Kind.COUNT_STAR, null), FileAggregate.countStar());
    }
}
