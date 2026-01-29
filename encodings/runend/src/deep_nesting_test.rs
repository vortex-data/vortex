// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test showing extremely deep nesting (8+ levels) with constrained array generation.

#[cfg(test)]
mod tests {
    use vortex_array::Array;
    use vortex_array::IntoArray;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::display::DisplayOptions;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_dtype::Nullability;
    use vortex_dtype::PType;
    use vortex_error::VortexExpect;
    use vortex_scalar::PValue;
    use vortex_sequence::SequenceArray;

    use crate::RunEndArray;

    /// Build an 8+ level deep nested array structure using only:
    /// - RunEndArray
    /// - DictArray
    /// - SequenceArray
    /// - PrimitiveArray
    ///
    /// Structure:
    /// Level 1: RunEndArray
    /// Level 2: ├─ ends: SequenceArray
    /// Level 2: └─ values: RunEndArray
    /// Level 3:    ├─ ends: SequenceArray
    /// Level 3:    └─ values: DictArray
    /// Level 4:       ├─ codes: RunEndArray
    /// Level 5:       │  ├─ ends: SequenceArray
    /// Level 5:       │  └─ values: DictArray
    /// Level 6:       │     ├─ codes: SequenceArray
    /// Level 6:       │     └─ values: RunEndArray
    /// Level 7:       │        ├─ ends: SequenceArray
    /// Level 7:       │        └─ values: DictArray
    /// Level 8:       │           ├─ codes: SequenceArray
    /// Level 8:       │           └─ values: Primitive (deepest!)
    /// Level 4:       └─ values: Primitive
    #[test]
    fn test_8_level_deep_nested_array() {
        // ========================================
        // Build from the bottom up (Level 8 first)
        // ========================================

        // Level 8: Deepest primitive (dict values)
        let level8_values: Vec<i32> = vec![1000, 2000, 3000, 4000];
        let level8_prim =
            PrimitiveArray::new(Buffer::copy_from(level8_values), Validity::NonNullable);

        // Level 8: SequenceArray for codes [0, 1, 2, 3]
        let level8_codes = SequenceArray::new(
            PValue::U8(0),
            PValue::U8(1),
            PType::U8,
            Nullability::NonNullable,
            4,
        )
        .vortex_expect("level8 codes");

        // Level 7: DictArray wrapping level 8
        let level7_dict = DictArray::try_new(level8_codes.into_array(), level8_prim.into_array())
            .vortex_expect("level7 dict");

        // Level 7: SequenceArray for ends [1, 2, 3, 4]
        let level7_ends = SequenceArray::new(
            PValue::U32(1),
            PValue::U32(1),
            PType::U32,
            Nullability::NonNullable,
            4,
        )
        .vortex_expect("level7 ends");

        // Level 6: RunEndArray wrapping level 7
        let level6_runend = RunEndArray::try_new(level7_ends.into_array(), level7_dict.into_array())
            .vortex_expect("level6 runend");

        // Level 6: SequenceArray for codes [0, 1, 2, 3]
        let level6_codes = SequenceArray::new(
            PValue::U8(0),
            PValue::U8(1),
            PType::U8,
            Nullability::NonNullable,
            4,
        )
        .vortex_expect("level6 codes");

        // Level 5: DictArray wrapping level 6
        let level5_dict =
            DictArray::try_new(level6_codes.into_array(), level6_runend.into_array())
                .vortex_expect("level5 dict");

        // Level 5: SequenceArray for ends [1, 2, 3, 4]
        let level5_ends = SequenceArray::new(
            PValue::U32(1),
            PValue::U32(1),
            PType::U32,
            Nullability::NonNullable,
            4,
        )
        .vortex_expect("level5 ends");

        // Level 4: RunEndArray wrapping level 5
        let level4_runend = RunEndArray::try_new(level5_ends.into_array(), level5_dict.into_array())
            .vortex_expect("level4 runend");

        // Level 3: Need unsigned codes for DictArray, create a sequence
        let level3_codes = SequenceArray::new(
            PValue::U8(0),
            PValue::U8(1),
            PType::U8,
            Nullability::NonNullable,
            4,
        )
        .vortex_expect("level3 codes");

        // Level 3: DictArray with sequence codes and level4_runend as values (to keep nesting)
        let level3_dict =
            DictArray::try_new(level3_codes.into_array(), level4_runend.into_array())
                .vortex_expect("level3 dict");

        // Level 3: SequenceArray for ends [1, 2, 3, 4]
        let level3_ends = SequenceArray::new(
            PValue::U32(1),
            PValue::U32(1),
            PType::U32,
            Nullability::NonNullable,
            4,
        )
        .vortex_expect("level3 ends");

        // Level 2: RunEndArray wrapping level 3
        let level2_runend = RunEndArray::try_new(level3_ends.into_array(), level3_dict.into_array())
            .vortex_expect("level2 runend");

        // Level 2: SequenceArray for outer ends [1, 2, 3, 4]
        let level2_ends = SequenceArray::new(
            PValue::U32(1),
            PValue::U32(1),
            PType::U32,
            Nullability::NonNullable,
            4,
        )
        .vortex_expect("level2 ends");

        // Level 1: Outer RunEndArray
        let level1_runend = RunEndArray::try_new(level2_ends.into_array(), level2_runend.into_array())
            .vortex_expect("level1 runend");

        // ========================================
        // Display the deeply nested structure
        // ========================================
        println!("\n=== 8-Level Deep Nested Array ===");
        println!(
            "Tree:\n{}",
            level1_runend.display_as(DisplayOptions::TreeDisplay)
        );

        // Count the nesting depth
        fn count_depth(array: &dyn Array, current: usize) -> usize {
            let child_depths: Vec<usize> = array
                .children()
                .iter()
                .map(|c| count_depth(c.as_ref(), current + 1))
                .collect();
            child_depths.into_iter().max().unwrap_or(current)
        }

        let depth = count_depth(level1_runend.as_ref(), 1);
        println!("\nMaximum nesting depth: {}", depth);
        println!("Total length: {}", level1_runend.len());

        assert!(depth >= 8, "Should have at least 8 levels of nesting, got {}", depth);
    }

    /// Simpler test: 8 levels using alternating RunEndArray and DictArray
    #[test]
    fn test_8_level_alternating_pattern() {
        // Level 8 (deepest): Primitive
        let base_data: Vec<u32> = vec![10, 20, 30, 40];
        let level8 = PrimitiveArray::new(Buffer::copy_from(base_data), Validity::NonNullable);
        println!("Level 8 (Primitive): len={}", level8.len());

        // Level 7: DictArray (codes=sequence, values=level8)
        let codes7 = SequenceArray::new(
            PValue::U8(0),
            PValue::U8(1),
            PType::U8,
            Nullability::NonNullable,
            4,
        )
        .vortex_expect("codes7");
        let level7 =
            DictArray::try_new(codes7.into_array(), level8.into_array()).vortex_expect("dict7");
        println!("Level 7 (Dict): len={}", level7.len());

        // Level 6: RunEndArray (ends=sequence, values=level7)
        let ends6 = SequenceArray::new(
            PValue::U32(1),
            PValue::U32(1),
            PType::U32,
            Nullability::NonNullable,
            4,
        )
        .vortex_expect("ends6");
        let level6 =
            RunEndArray::try_new(ends6.into_array(), level7.into_array()).vortex_expect("runend6");
        println!("Level 6 (RunEnd): len={}", level6.len());

        // Level 5: DictArray
        let codes5 = SequenceArray::new(
            PValue::U8(0),
            PValue::U8(1),
            PType::U8,
            Nullability::NonNullable,
            4,
        )
        .vortex_expect("codes5");
        let level5 =
            DictArray::try_new(codes5.into_array(), level6.into_array()).vortex_expect("dict5");
        println!("Level 5 (Dict): len={}", level5.len());

        // Level 4: RunEndArray
        let ends4 = SequenceArray::new(
            PValue::U32(1),
            PValue::U32(1),
            PType::U32,
            Nullability::NonNullable,
            4,
        )
        .vortex_expect("ends4");
        let level4 =
            RunEndArray::try_new(ends4.into_array(), level5.into_array()).vortex_expect("runend4");
        println!("Level 4 (RunEnd): len={}", level4.len());

        // Level 3: DictArray
        let codes3 = SequenceArray::new(
            PValue::U8(0),
            PValue::U8(1),
            PType::U8,
            Nullability::NonNullable,
            4,
        )
        .vortex_expect("codes3");
        let level3 =
            DictArray::try_new(codes3.into_array(), level4.into_array()).vortex_expect("dict3");
        println!("Level 3 (Dict): len={}", level3.len());

        // Level 2: RunEndArray
        let ends2 = SequenceArray::new(
            PValue::U32(1),
            PValue::U32(1),
            PType::U32,
            Nullability::NonNullable,
            4,
        )
        .vortex_expect("ends2");
        let level2 =
            RunEndArray::try_new(ends2.into_array(), level3.into_array()).vortex_expect("runend2");
        println!("Level 2 (RunEnd): len={}", level2.len());

        // Level 1: DictArray (outermost)
        let codes1 = SequenceArray::new(
            PValue::U8(0),
            PValue::U8(1),
            PType::U8,
            Nullability::NonNullable,
            4,
        )
        .vortex_expect("codes1");
        let level1 =
            DictArray::try_new(codes1.into_array(), level2.into_array()).vortex_expect("dict1");
        println!("Level 1 (Dict): len={}", level1.len());

        println!("\n=== 8-Level Alternating Pattern ===");
        println!(
            "Tree:\n{}",
            level1.display_as(DisplayOptions::TreeDisplay)
        );

        // Count depth
        fn count_depth(array: &dyn Array, current: usize) -> usize {
            let child_depths: Vec<usize> = array
                .children()
                .iter()
                .map(|c| count_depth(c.as_ref(), current + 1))
                .collect();
            child_depths.into_iter().max().unwrap_or(current)
        }

        let depth = count_depth(level1.as_ref(), 1);
        println!("\nMaximum nesting depth: {}", depth);
        assert!(depth >= 8, "Should have at least 8 levels of nesting, got {}", depth);
    }
}
