// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.spark;

import static org.junit.jupiter.api.Assertions.*;

import dev.vortex.api.DType;
import dev.vortex.jni.NativeLoader;
import org.apache.spark.sql.types.ArrayType;
import org.apache.spark.sql.types.DataTypes;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

public final class SparkTypesTest {

    @BeforeAll
    public static void loadLibrary() {
        NativeLoader.loadJni();
    }

    @Test
    @DisplayName("toDataType should convert FIXED_SIZE_LIST to Spark ArrayType")
    public void testFixedSizeListToDataType() {
        var elementType = DType.newInt(false);
        var fslType = DType.newFixedSizeList(elementType, 3, true);
        var sparkType = SparkTypes.toDataType(fslType);
        assertInstanceOf(ArrayType.class, sparkType);
        ArrayType arrayType = (ArrayType) sparkType;
        assertEquals(DataTypes.IntegerType, arrayType.elementType());
    }

    @Test
    @DisplayName("toDataType should convert LIST to Spark ArrayType")
    public void testListToDataType() {
        var elementType = DType.newDouble(false);
        var listType = DType.newList(elementType, true);
        var sparkType = SparkTypes.toDataType(listType);
        assertInstanceOf(ArrayType.class, sparkType);
        ArrayType arrayType = (ArrayType) sparkType;
        assertEquals(DataTypes.DoubleType, arrayType.elementType());
    }
}
