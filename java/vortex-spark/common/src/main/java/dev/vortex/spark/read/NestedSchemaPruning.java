// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import static com.google.common.base.Preconditions.checkArgument;

import dev.vortex.api.Expression;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.ArrowType;
import dev.vortex.relocated.org.apache.arrow.vector.types.pojo.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeMap;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

/**
 * Pushes Spark's nested schema pruning down to the native scan, and resolves the read schema against each file.
 *
 * <p>Columns and struct fields resolve by name under Spark's case sensitivity, so a file can store a struct's fields in
 * another order, in another case, or without some nullable fields, and still read correctly.
 *
 * <p>Only struct fields are pruned and resolved by name. Structs inside arrays and maps, and extension types that
 * export as Arrow structs such as Variant, are read whole and by position, since projecting each element needs a
 * per-element expression the native scan does not expose yet.
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
     * How the fields of read struct {@code type} come out of the struct the scan returns: from position
     * {@code slots[i]}, or as nulls where the file does not carry the field and the slot is -1. {@code children} holds
     * the shape of each struct field that needs one, and is {@code null} elsewhere.
     */
    static final class Shape {
        private final StructType type;
        private final int[] slots;
        private final Shape[] children;

        private Shape(StructType type, int[] slots, Shape[] children) {
            this.type = type;
            this.slots = slots;
            this.children = children;
        }

        /** Lays out {@code columns}, the fields of a struct the scan returned, as the fields of {@link #type}. */
        VortexArrowColumnVector[] layout(VortexArrowColumnVector[] columns, int rowCount) {
            StructField[] fields = type.fields();
            VortexArrowColumnVector[] laidOut = new VortexArrowColumnVector[fields.length];
            for (int i = 0; i < fields.length; i++) {
                if (slots[i] < 0) {
                    laidOut[i] = VortexArrowColumnVector.nulls(fields[i].dataType(), rowCount);
                } else if (children[i] != null) {
                    VortexArrowColumnVector struct = columns[slots[i]];
                    laidOut[i] =
                            struct.withFields(fields[i].dataType(), children[i].layout(struct.childColumns, rowCount));
                } else {
                    laidOut[i] = columns[slots[i]];
                }
            }
            return laidOut;
        }
    }

    /** A native projection, and the {@link Shape} of its result, or {@code null} if it reads as returned. */
    record Projection(Expression expression, Shape shape) {}

    /**
     * Builds the native projection that reads {@code readSchema} from {@code fileName}, whose columns are
     * {@code fileColumns}.
     *
     * <p>Fields resolve by name under Spark's case sensitivity, at every level, and are read under the file's names, in
     * the order Spark reads them and only as far down as Spark reads them. A nullable field the file does not carry is
     * left out and filled in with nulls through {@link Projection#shape}.
     *
     * <p>More top-level columns follow the fields of {@code readSchema}: one named {@code extraNames[i]} that
     * {@code extraColumns[i]} computes for each {@code i}. They come after the columns that {@link Projection#shape}
     * lays out.
     */
    static Projection projection(
            StructType readSchema,
            List<Field> fileColumns,
            boolean caseSensitive,
            String fileName,
            List<String> extraNames,
            List<Expression> extraColumns) {
        Expression root = Expression.root();
        Projection projection =
                struct(root, false, readSchema, fileColumns, caseSensitive, fileName, "", extraNames, extraColumns);
        if (projection.expression() != null) {
            return projection;
        }
        String[] names = fileColumns.stream().map(Field::getName).toArray(String[]::new);
        return new Projection(Expression.select(names, root), projection.shape());
    }

    /** Returns the projection of {@code struct}, with a {@code null} expression when the file stores it as read. */
    private static Projection struct(
            Expression struct,
            boolean nullable,
            StructType read,
            List<Field> file,
            boolean caseSensitive,
            String fileName,
            String prefix,
            List<String> extraNames,
            List<Expression> extraColumns) {
        StructField[] fields = read.fields();
        FieldLookup lookup = new FieldLookup(file, caseSensitive);
        List<String> names = new ArrayList<>(fields.length);
        List<Expression> children = new ArrayList<>(fields.length);
        int[] slots = new int[fields.length];
        Shape[] shapes = new Shape[fields.length];
        boolean reshaped = fields.length != file.size();
        boolean packed = false;
        boolean filled = false;
        for (int i = 0; i < fields.length; i++) {
            String path = prefix + fields[i].name();
            int index = lookup.indexOf(fields[i].name(), fileName, prefix);
            if (index < 0) {
                checkArgument(
                        fields[i].nullable(),
                        "%s does not carry the non-nullable column %s that the query requires",
                        fileName,
                        path);
                slots[i] = -1;
                reshaped = true;
                filled = true;
                continue;
            }
            reshaped |= index != names.size();
            slots[i] = names.size();
            Field fileField = file.get(index);
            Expression child = Expression.getItem(fileField.getName(), struct);

            if (fields[i].dataType() instanceof StructType) {
                checkArgument(
                        fileField.getType() instanceof ArrowType.Struct
                                || fileField.getType() instanceof ArrowType.ExtensionType,
                        "%s stores %s as %s, but the query reads it as a struct",
                        fileName,
                        path,
                        fileField.getType());
            }

            // An extension type such as Variant exports as an Arrow struct but is not a native struct.
            if (fields[i].dataType() instanceof StructType readStruct
                    && fileField.getType() instanceof ArrowType.Struct
                    && !fileField.getMetadata().containsKey(ArrowType.ExtensionType.EXTENSION_METADATA_KEY_NAME)) {
                Projection nested = struct(
                        child,
                        fileField.isNullable(),
                        readStruct,
                        fileField.getChildren(),
                        caseSensitive,
                        fileName,
                        path + ".",
                        List.of(),
                        List.of());
                if (nested.expression() != null) {
                    child = nested.expression();
                    packed = true;
                }
                shapes[i] = nested.shape();
                filled |= nested.shape() != null;
            }
            names.add(fileField.getName());
            children.add(child);
        }
        Shape shape = filled ? new Shape(read, slots, shapes) : null;
        names.addAll(extraNames);
        children.addAll(extraColumns);
        packed |= !extraNames.isEmpty();
        String[] fileNames = names.toArray(new String[0]);
        // `select` keeps the struct's own nulls, so use it when no child needs packing. An empty one loses them.
        if (!packed && !(nullable && names.isEmpty())) {
            return new Projection(reshaped ? Expression.select(fileNames, struct) : null, shape);
        }
        Expression pack = Expression.pack(fileNames, children.toArray(new Expression[0]), false);
        // `pack` builds a struct with no nulls, so carry over the source struct's nulls.
        return new Projection(nullable ? Expression.mask(pack, Expression.isNotNull(struct)) : pack, shape);
    }

    /**
     * Finds the fields of one file struct by name under Spark's case sensitivity, with one map lookup per field, so
     * resolving a wide schema stays linear in its width.
     */
    private static final class FieldLookup {
        private final List<Field> fields;
        private final Map<String, Integer> indices;

        /** The second field that matches a name, for names that more than one field matches. */
        private final Map<String, Integer> duplicates;

        FieldLookup(List<Field> fields, boolean caseSensitive) {
            this.fields = fields;
            this.indices = caseSensitive ? new HashMap<>() : new TreeMap<>(String.CASE_INSENSITIVE_ORDER);
            this.duplicates = caseSensitive ? new HashMap<>() : new TreeMap<>(String.CASE_INSENSITIVE_ORDER);

            for (int i = 0; i < fields.size(); i++) {
                String name = fields.get(i).getName();
                if (indices.putIfAbsent(name, i) != null) {
                    duplicates.putIfAbsent(name, i);
                }
            }
        }

        /** Returns the position of the field {@code name} resolves to, or -1 if the struct does not carry it. */
        int indexOf(String name, String fileName, String prefix) {
            Integer index = indices.get(name);
            if (index == null) {
                return -1;
            }

            Integer duplicate = duplicates.get(name);
            if (duplicate != null) {
                throw new IllegalArgumentException(String.format(
                        Locale.ROOT,
                        "Ambiguous columns %s and %s in %s under Spark's case sensitivity setting",
                        prefix + fields.get(index).getName(),
                        prefix + fields.get(duplicate).getName(),
                        fileName));
            }

            return index;
        }
    }
}
