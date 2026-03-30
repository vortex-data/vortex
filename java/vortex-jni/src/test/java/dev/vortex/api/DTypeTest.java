// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

package dev.vortex.api;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

public final class DTypeTest {

    @Test
    public void testNewFixedSizeListNonNullable() {
        var elementType = DType.newInt(false);
        var fslType = DType.newFixedSizeList(elementType, 3, false);
        assertEquals(DType.Variant.FIXED_SIZE_LIST, fslType.getVariant());
        assertFalse(fslType.isNullable());
        assertEquals(3, fslType.getFixedSizeListSize());

        var innerType = fslType.getElementType();
        assertEquals(DType.Variant.PRIMITIVE_I32, innerType.getVariant());
    }

    @Test
    public void testNewFixedSizeListNullable() {
        var elementType = DType.newUtf8(true);
        var fslType = DType.newFixedSizeList(elementType, 5, true);
        assertEquals(DType.Variant.FIXED_SIZE_LIST, fslType.getVariant());
        assertTrue(fslType.isNullable());
        assertEquals(5, fslType.getFixedSizeListSize());

        var innerType = fslType.getElementType();
        assertEquals(DType.Variant.UTF8, innerType.getVariant());
    }

    @Test
    public void testNewListGetElementType() {
        var elementType = DType.newDouble(false);
        var listType = DType.newList(elementType, false);
        assertEquals(DType.Variant.LIST, listType.getVariant());

        var innerType = listType.getElementType();
        assertEquals(DType.Variant.PRIMITIVE_F64, innerType.getVariant());
    }

    @Test
    public void testNestedFixedSizeList() {
        var innerElement = DType.newLong(false);
        var innerFsl = DType.newFixedSizeList(innerElement, 2, false);
        var outerFsl = DType.newFixedSizeList(innerFsl, 4, true);
        assertEquals(DType.Variant.FIXED_SIZE_LIST, outerFsl.getVariant());
        assertTrue(outerFsl.isNullable());
        assertEquals(4, outerFsl.getFixedSizeListSize());

        var inner = outerFsl.getElementType();
        assertEquals(DType.Variant.FIXED_SIZE_LIST, inner.getVariant());
    }

    @Test
    public void testFixedSizeListInStruct() {
        var elementType = DType.newFloat(false);
        var fslType = DType.newFixedSizeList(elementType, 3, false);
        var structType = DType.newStruct(
                new String[] {"id", "embedding"},
                new DType[] {DType.newInt(false), fslType},
                false);
        assertEquals(DType.Variant.STRUCT, structType.getVariant());

        var fieldTypes = structType.getFieldTypes();
        assertEquals(2, fieldTypes.size());

        var embeddingType = fieldTypes.get(1);
        assertEquals(DType.Variant.FIXED_SIZE_LIST, embeddingType.getVariant());
    }
}
