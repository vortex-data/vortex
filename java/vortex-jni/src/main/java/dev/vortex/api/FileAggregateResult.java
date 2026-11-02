// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import com.google.common.base.Preconditions;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.VectorSchemaRoot;

/**
 * The answers to a list of {@link FileAggregate}s over one Vortex file: a one-row Arrow batch with one column per
 * aggregate, in request order.
 *
 * <p>An aggregate that was not answered holds a null placeholder; {@link #answered(int)} tells it apart from a null
 * answer, such as the minimum of a column of nothing but nulls. The caller owns the batch and must close the result.
 */
public final class FileAggregateResult implements AutoCloseable {
    private final VectorSchemaRoot root;
    private final boolean[] answered;

    FileAggregateResult(VectorSchemaRoot root, boolean[] answered) {
        Preconditions.checkArgument(
                root.getFieldVectors().size() == answered.length, "one answer flag is required per aggregate");
        Preconditions.checkArgument(root.getRowCount() == 1, "aggregates answer with exactly one row");
        this.root = root;
        this.answered = answered;
    }

    /** Builds a result. Used by the JNI layer, which imports the native batch. */
    public static FileAggregateResult of(VectorSchemaRoot root, boolean[] answered) {
        return new FileAggregateResult(root, answered.clone());
    }

    /** The number of aggregates. */
    public int size() {
        return answered.length;
    }

    /** Whether aggregate {@code index} was answered, rather than left as a null placeholder. */
    public boolean answered(int index) {
        return answered[index];
    }

    /** Whether every aggregate was answered. */
    public boolean allAnswered() {
        for (boolean value : answered) {
            if (!value) {
                return false;
            }
        }
        return true;
    }

    /** The one-row vector holding the answer of aggregate {@code index}. */
    public FieldVector vector(int index) {
        return root.getVector(index);
    }

    /** The one-row batch of every answer. */
    public VectorSchemaRoot root() {
        return root;
    }

    @Override
    public void close() {
        root.close();
    }
}
