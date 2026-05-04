// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark.read;

import com.google.common.base.Splitter;
import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import com.google.common.primitives.Longs;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;
import org.apache.spark.sql.execution.vectorized.ConstantColumnVector;
import org.apache.spark.sql.types.BooleanType;
import org.apache.spark.sql.types.ByteType;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.DateType;
import org.apache.spark.sql.types.DoubleType;
import org.apache.spark.sql.types.FloatType;
import org.apache.spark.sql.types.IntegerType;
import org.apache.spark.sql.types.LongType;
import org.apache.spark.sql.types.ShortType;
import org.apache.spark.sql.types.StringType;
import org.apache.spark.sql.types.TimestampNTZType;
import org.apache.spark.sql.types.TimestampType;
import org.apache.spark.unsafe.types.UTF8String;

/** Utilities for discovering and materializing Hive-style partition columns from file paths. */
public final class PartitionPathUtils {
    private static final String HIVE_DEFAULT_PARTITION = "__HIVE_DEFAULT_PARTITION__";
    private static final Splitter PATH_SPLITTER = Splitter.on('/');

    private PartitionPathUtils() {}

    /**
     * Parses Hive-style {@code key=value} segments from a file path.
     *
     * @return an ordered map of partition column names to their string values
     */
    public static Map<String, String> parsePartitionValues(String filePath) {
        LinkedHashMap<String, String> values = new LinkedHashMap<>();
        for (String segment : PATH_SPLITTER.split(filePath)) {
            int eqIdx = segment.indexOf('=');
            if (eqIdx > 0 && eqIdx < segment.length() - 1) {
                String key = URLDecoder.decode(segment.substring(0, eqIdx), StandardCharsets.UTF_8);
                String val = URLDecoder.decode(segment.substring(eqIdx + 1), StandardCharsets.UTF_8);
                values.put(key, val);
            }
        }
        return values;
    }

    /**
     * Infers a Spark {@link DataType} from a partition value string. Tries integer, long, double, boolean, and falls
     * back to string.
     */
    public static DataType inferPartitionColumnType(String value) {
        if (value == null || HIVE_DEFAULT_PARTITION.equals(value)) {
            return DataTypes.StringType;
        }
        if (Ints.tryParse(value) != null) {
            return DataTypes.IntegerType;
        }
        if (Longs.tryParse(value) != null) {
            return DataTypes.LongType;
        }
        if (Doubles.tryParse(value) != null) {
            return DataTypes.DoubleType;
        }
        if ("true".equalsIgnoreCase(value) || "false".equalsIgnoreCase(value)) {
            return DataTypes.BooleanType;
        }
        return DataTypes.StringType;
    }

    /**
     * Creates a Spark {@link ConstantColumnVector} populated with the given partition value, parsed according to the
     * target {@link DataType}.
     */
    public static ConstantColumnVector createConstantVector(int numRows, DataType type, String value) {
        ConstantColumnVector vec = new ConstantColumnVector(numRows, type);
        if (value == null || HIVE_DEFAULT_PARTITION.equals(value)) {
            vec.setNull();
            return vec;
        }
        vec.setNotNull();
        if (type instanceof StringType) {
            vec.setUtf8String(UTF8String.fromString(value));
        } else if (type instanceof IntegerType || type instanceof DateType) {
            vec.setInt(Integer.parseInt(value));
        } else if (type instanceof LongType || type instanceof TimestampType || type instanceof TimestampNTZType) {
            vec.setLong(Long.parseLong(value));
        } else if (type instanceof ShortType) {
            vec.setShort(Short.parseShort(value));
        } else if (type instanceof ByteType) {
            vec.setByte(Byte.parseByte(value));
        } else if (type instanceof BooleanType) {
            vec.setBoolean(Boolean.parseBoolean(value));
        } else if (type instanceof FloatType) {
            vec.setFloat(Float.parseFloat(value));
        } else if (type instanceof DoubleType) {
            vec.setDouble(Double.parseDouble(value));
        } else {
            vec.setUtf8String(UTF8String.fromString(value));
        }
        return vec;
    }
}
