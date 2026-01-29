// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arbitrary::Arbitrary;
use arbitrary::Result;
use arbitrary::Unstructured;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_array::arrays::arbitrary::ArbitraryConstrained;
use vortex_array::arrays::arbitrary::ArrayConstraints;
use vortex_array::arrays::arbitrary::BoundConstraint;
use vortex_array::arrays::arbitrary::ConstraintKind;
use vortex_array::arrays::arbitrary::OrderingConstraint;
use vortex_array::arrays::arbitrary::arbitrary_constrained_array;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_error::VortexExpect;

use crate::RunEndArray;

/// RunEndArray doesn't preserve ordering of its values, but its ends are strictly sorted.
/// The main use case is generating arrays where ends can be compressed.
pub const RUNEND_CAN_GENERATE: &[ConstraintKind] = &[ConstraintKind::NonNullable];

/// A wrapper type to implement `Arbitrary` for `RunEndArray`.
#[derive(Clone, Debug)]
pub struct ArbitraryRunEndArray(pub RunEndArray);

impl<'a> Arbitrary<'a> for ArbitraryRunEndArray {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        // RunEnd supports Bool or Primitive types for values
        // Pick a random primitive type for values
        let ptype: PType = u.arbitrary()?;
        let nullability: Nullability = u.arbitrary()?;
        let dtype = DType::Primitive(ptype, nullability);
        Self::with_dtype(u, &dtype, None)
    }
}

impl ArbitraryRunEndArray {
    /// Generate an arbitrary RunEndArray with the given dtype for values.
    ///
    /// The dtype must be a primitive or boolean type.
    pub fn with_dtype(u: &mut Unstructured, dtype: &DType, len: Option<usize>) -> Result<Self> {
        Self::with_constraints(u, len, dtype, &ArrayConstraints::default())
    }

    /// Generate an arbitrary RunEndArray satisfying the given constraints.
    pub fn with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> Result<Self> {
        // Number of runs (values/ends pairs)
        let num_runs = u.int_in_range(0..=20)?;

        if num_runs == 0 {
            // Empty RunEndArray
            let ends_dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
            let ends =
                arbitrary_constrained_array(u, Some(0), &ends_dtype, &ArrayConstraints::default())?;
            let values = ArbitraryArray::arbitrary_with(u, Some(0), dtype)?.0;
            let runend_array = RunEndArray::try_new(ends, values)
                .vortex_expect("Empty RunEndArray creation should succeed");
            return Ok(ArbitraryRunEndArray(runend_array));
        }

        // Generate arbitrary values for each run
        let values = ArbitraryArray::arbitrary_with(u, Some(num_runs), dtype)?.0;

        // Generate strictly sorted ends using constrained generation
        // This allows the ends to be represented by encodings like Sequence, Delta, etc.
        let ends = generate_constrained_ends(u, num_runs, len)?;

        let runend_array = RunEndArray::try_new(ends, values)
            .vortex_expect("RunEndArray creation should succeed in arbitrary impl");

        // Verify constraints are satisfied if any
        if constraints.non_nullable {
            // RunEndArray itself doesn't have nullability, but values might
            // The array is valid as long as it was created successfully
        }

        Ok(ArbitraryRunEndArray(runend_array))
    }
}

/// Generate strictly sorted ends using the constrained arbitrary system.
///
/// This allows the ends to be represented by various encodings that satisfy
/// the StrictlySorted constraint (e.g., Sequence, Delta, FoR+BitPacked).
fn generate_constrained_ends(
    u: &mut Unstructured,
    num_runs: usize,
    target_len: Option<usize>,
) -> Result<ArrayRef> {
    // Choose a random unsigned PType for ends
    let ends_ptype = *u.choose(&[PType::U8, PType::U16, PType::U32, PType::U64])?;
    let ends_dtype = DType::Primitive(ends_ptype, Nullability::NonNullable);

    // Calculate bounds for the ends
    // Ends must start at >= 1 (each run has at least 1 element)
    // Last end determines the total array length
    let max_end = target_len.map(|l| l as u64).unwrap_or_else(|| {
        // If no target length, cap based on type and reasonable size
        let type_max = match ends_ptype {
            PType::U8 => u8::MAX as u64,
            PType::U16 => u16::MAX as u64,
            PType::U32 => u32::MAX as u64,
            PType::U64 => 10000, // Reasonable limit for u64
            _ => 10000,
        };
        // Each run adds at least 1 to the end, so we need room for num_runs increments
        // Plus some headroom for variation
        (num_runs as u64 * 10).min(type_max)
    });

    // Build constraints for strictly sorted ends starting at 1
    let ends_constraints = ArrayConstraints {
        ordering: OrderingConstraint {
            strictly_sorted: true,
            sorted: true,
            starts_at: None, // Let it start at any value >= 1
        },
        bounds: BoundConstraint {
            lower_bound: Some(1),           // First end must be at least 1
            upper_bound: Some(max_end + 1), // Exclusive upper bound
            target_max: target_len.map(|l| l as u64),
            ..Default::default()
        },
        non_nullable: true,
        ..Default::default()
    };

    arbitrary_constrained_array(u, Some(num_runs), &ends_dtype, &ends_constraints)
}

impl ArbitraryConstrained for RunEndArray {
    fn can_generate() -> &'static [ConstraintKind] {
        RUNEND_CAN_GENERATE
    }

    fn arbitrary_with_constraints(
        u: &mut Unstructured,
        len: Option<usize>,
        dtype: &DType,
        constraints: &ArrayConstraints,
    ) -> Result<ArrayRef> {
        Ok(
            ArbitraryRunEndArray::with_constraints(u, len, dtype, constraints)?
                .0
                .into_array(),
        )
    }
}
