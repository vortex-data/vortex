// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import dev.vortex.api.Expression;
import java.util.ArrayList;
import java.util.List;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

/**
 * Pushes Spark's nested schema pruning down to the native scan.
 *
 * <p>Only struct fields are pruned. Fields inside arrays and maps keep their full table type, since projecting each
 * element needs a per-element expression the native scan does not expose yet.
 */
final class NestedSchemaPruning {
    private NestedSchemaPruning() {}

    /**
     * Returns {@code tableType} pruned to the struct fields present in {@code requiredType}, in table order. Any type
     * that is not a struct on both sides keeps its table type.
     */
    static DataType prune(DataType tableType, DataType requiredType) {
        if (!(tableType instanceof StructType table) || !(requiredType instanceof StructType required)) {
            return tableType;
        }
        List<StructField> fields = new ArrayList<>();
        for (StructField field : table.fields()) {
            if (required.getFieldIndex(field.name()).isDefined()) {
                DataType prunedType =
                        prune(field.dataType(), required.apply(field.name()).dataType());
                fields.add(new StructField(field.name(), prunedType, field.nullable(), field.metadata()));
            }
        }
        return fields.isEmpty() ? tableType : new StructType(fields.toArray(new StructField[0]));
    }

    /**
     * Builds the native projection that returns {@code readSchema}, where each field is either its type in
     * {@code tableSchema} or a {@link #prune pruned} version of it.
     *
     * <p>When nothing is pruned this is a plain {@code select} of the top-level columns.
     */
    static Expression projection(StructType readSchema, StructType tableSchema) {
        return structProjection(Expression.root(), false, readSchema, tableSchema);
    }

    private static Expression structProjection(Expression struct, boolean nullable, StructType read, StructType table) {
        String[] names = read.fieldNames();
        // `select` keeps the struct's own nulls, so use it when no child needs pruning.
        if (!isPruned(read, table)) {
            return Expression.select(names, struct);
        }
        Expression[] children = new Expression[names.length];
        for (int i = 0; i < names.length; i++) {
            Expression field = Expression.getItem(names[i], struct);
            StructField readField = read.fields()[i];
            DataType tableType = table.apply(names[i]).dataType();
            children[i] = readField.dataType().equals(tableType)
                    ? field
                    : structProjection(
                            field, readField.nullable(), (StructType) readField.dataType(), (StructType) tableType);
        }
        Expression packed = Expression.pack(names, children, false);
        // `pack` builds a struct with no nulls, so carry over the source struct's nulls.
        return nullable ? Expression.mask(packed, Expression.isNotNull(struct)) : packed;
    }

    private static boolean isPruned(StructType read, StructType table) {
        for (StructField field : read.fields()) {
            if (!field.dataType().equals(table.apply(field.name()).dataType())) {
                return true;
            }
        }
        return false;
    }
}
