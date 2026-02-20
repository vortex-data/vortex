// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Test showing deeply nested constrained array generation.

#[cfg(test)]
#[allow(clippy::cast_possible_truncation, clippy::use_debug, clippy::len_zero)]
mod tests {
    use arbitrary::Unstructured;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::arbitrary::ArbitraryArray;
    use vortex_array::arrays::arbitrary::ArrayConstraints;
    use vortex_array::arrays::arbitrary::BoundConstraint;
    use vortex_array::display::DisplayOptions;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::PValue;
    use vortex_buffer::Buffer;
    use vortex_error::VortexExpect;
    use vortex_sequence::SequenceArray;

    use crate::RunEndArray;

    /// Create a deeply nested RunEndArray where:
    /// - The outer array is RunEndArray
    /// - The ends are a SequenceArray (sorted by construction)
    /// - The values are arbitrary primitives
    #[test]
    fn test_deeply_nested_runend_with_sequence_ends() {
        let seed: Vec<u8> = (100..3100).map(|i| (i % 256) as u8).collect();
        let mut u = Unstructured::new(&seed);

        // Create a SequenceArray for ends: base=1, multiplier=3 => [1, 4, 7, 10, 13, ...]
        // This is STRICTLY SORTED by construction!
        let num_runs = 8;
        let base = 1u64;
        let multiplier = 3u64;

        let ends_sequence = SequenceArray::new(
            PValue::U64(base),
            PValue::U64(multiplier),
            PType::U64,
            Nullability::NonNullable,
            num_runs,
        )
        .vortex_expect("SequenceArray creation should succeed");

        // Generate arbitrary values for each run
        let values_dtype =
            vortex_array::dtype::DType::Primitive(PType::I32, Nullability::NonNullable);
        let values = ArbitraryArray::arbitrary_with(&mut u, Some(num_runs), &values_dtype)
            .unwrap()
            .0;

        // Create the RunEndArray with SequenceArray ends
        let runend = RunEndArray::try_new(ends_sequence.into_array(), values)
            .vortex_expect("RunEndArray creation should succeed");

        println!("\n=== Deeply Nested: RunEndArray with SequenceArray ends ===");
        println!(
            "Tree:\n{}",
            runend.display_as(DisplayOptions::TreeDisplay {
                buffers: false,
                metadata: false,
                stats: false
            })
        );
        println!(
            "\nEnds encoding: vortex.sequence (base={}, multiplier={})",
            base, multiplier
        );

        // Show the logical ends values
        let ends_primitive = runend.ends().to_primitive();
        println!("Ends values: {:?}", ends_primitive.as_slice::<u64>());

        // Verify they're strictly sorted
        let ends_slice = ends_primitive.as_slice::<u64>();
        for i in 1..ends_slice.len() {
            assert!(ends_slice[i] > ends_slice[i - 1], "Must be strictly sorted");
        }
        println!("✓ Ends are strictly sorted (via SequenceArray)!");

        // Show values
        let values_primitive = runend.values().to_primitive();
        println!("Values: {:?}", values_primitive.as_slice::<i32>());
        println!("Total length: {}", runend.len());
    }

    /// Show a triple-nested structure:
    /// RunEndArray -> DictArray(codes) -> Primitive (bounded)
    /// RunEndArray -> ends -> SequenceArray
    #[test]
    fn test_triple_nested_structure() {
        use vortex_array::arrays::DictArray;

        let seed: Vec<u8> = (200..3200).map(|i| (i % 256) as u8).collect();
        let mut u = Unstructured::new(&seed);

        // Layer 1: Create a small dictionary of unique values
        let dict_values = PrimitiveArray::new(
            Buffer::copy_from(vec![100i32, 200, 300, 400, 500]),
            vortex_array::validity::Validity::NonNullable,
        )
        .into_array();

        // Layer 2: Create bounded random codes (indices into dict)
        let num_elements = 20;
        let codes_constraints = ArrayConstraints {
            bounds: BoundConstraint {
                lower_bound: Some(0),
                upper_bound: Some(5), // dict has 5 elements
                ..Default::default()
            },
            non_nullable: true,
            ..Default::default()
        };
        let codes_dtype =
            vortex_array::dtype::DType::Primitive(PType::U8, Nullability::NonNullable);
        let codes = vortex_array::arrays::arbitrary::arbitrary_constrained_array(
            &mut u,
            Some(num_elements),
            &codes_dtype,
            &codes_constraints,
        )
        .unwrap();

        // Create DictArray
        let dict_array = DictArray::try_new(codes, dict_values).vortex_expect("DictArray creation");

        // Layer 3: Use DictArray as values in a RunEndArray
        // Create strictly sorted ends using SequenceArray
        let num_runs = 5;
        let ends = SequenceArray::new(
            PValue::U32(4), // base=4
            PValue::U32(4), // multiplier=4 => [4, 8, 12, 16, 20]
            PType::U32,
            Nullability::NonNullable,
            num_runs,
        )
        .vortex_expect("SequenceArray for ends");

        // Get first 5 elements of dict_array as values
        let values = dict_array.slice(0..num_runs).unwrap();

        let runend =
            RunEndArray::try_new(ends.into_array(), values).vortex_expect("RunEndArray creation");

        println!("\n=== Triple Nested Structure ===");
        println!("Structure: RunEndArray -> values:DictArray -> codes:Primitive");
        println!("           RunEndArray -> ends:SequenceArray");
        println!(
            "\nTree:\n{}",
            runend.display_as(DisplayOptions::TreeDisplay {
                buffers: false,
                metadata: false,
                stats: false
            })
        );
        println!(
            "\nEnds (SequenceArray base=4, mult=4): {:?}",
            runend.ends().to_primitive().as_slice::<u32>()
        );
        println!("Total length: {}", runend.len());
    }
}
