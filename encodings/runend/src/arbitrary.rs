// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arbitrary::Arbitrary;
use arbitrary::Result;
use arbitrary::Unstructured;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;

use crate::RunEndData;

/// A wrapper type to implement `Arbitrary` for `RunEndArray`.
#[derive(Clone, Debug)]
pub struct ArbitraryRunEndArray(pub RunEndData);

impl<'a> Arbitrary<'a> for ArbitraryRunEndArray {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        // Pick a random primitive type for values.
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
        // Number of runs (values/ends pairs)
        let num_runs = u.int_in_range(0..=20)?;

        if num_runs == 0 {
            // Empty RunEndArray
            let ends = PrimitiveArray::from_iter(Vec::<u64>::new()).into_array();
            let values = ArbitraryArray::arbitrary_with(u, Some(0), dtype)?.0;
            let runend_array = RunEndData::try_new(ends, values)
                .vortex_expect("Empty RunEndArray creation should succeed");
            return Ok(ArbitraryRunEndArray(runend_array));
        }

        // Generate arbitrary values for each run
        let values = ArbitraryArray::arbitrary_with(u, Some(num_runs), dtype)?.0;

        // Generate strictly increasing ends
        // Each end must be > previous end, and first end must be >= 1
        let ends = random_strictly_sorted_ends(u, num_runs, len)?;

        let runend_array = RunEndData::try_new(ends, values)
            .vortex_expect("RunEndArray creation should succeed in arbitrary impl");

        Ok(ArbitraryRunEndArray(runend_array))
    }
}

/// Generate a strictly sorted array of run ends.
///
/// Returns an array of `num_runs` strictly increasing unsigned integers.
/// If `target_len` is provided, the last end will be exactly that value.
fn random_strictly_sorted_ends(
    u: &mut Unstructured,
    num_runs: usize,
    target_len: Option<usize>,
) -> Result<vortex_array::ArrayRef> {
    // Choose a random unsigned PType for ends
    let ends_ptype = *u.choose(&[PType::U8, PType::U16, PType::U32, PType::U64])?;

    // Generate strictly increasing values
    // Start from 0, increment by at least 1 each time
    let mut ends: Vec<u64> = Vec::with_capacity(num_runs);
    let mut current: u64 = 0;

    for i in 0..num_runs {
        // Each run must have at least length 1, so increment by at least 1
        let increment = match (i == num_runs - 1, target_len) {
            (true, Some(target)) => {
                // Last element should reach target_len
                let target = target as u64;
                if target > current {
                    target - current
                } else {
                    1
                }
            }
            _ => {
                // Random increment between 1 and 10
                u.int_in_range(1..=10)?
            }
        };
        current += increment;
        ends.push(current);
    }

    // Convert to the chosen PType
    // The values are bounded: max is num_runs (20) * max_increment (10) = 200
    // This fits in all unsigned types
    let ends_array = match ends_ptype {
        PType::U8 => {
            let ends_typed: Vec<u8> = ends
                .iter()
                .map(|&e| u8::try_from(e).vortex_expect("end value fits in u8"))
                .collect();
            PrimitiveArray::new(Buffer::copy_from(ends_typed), Validity::NonNullable).into_array()
        }
        PType::U16 => {
            let ends_typed: Vec<u16> = ends
                .iter()
                .map(|&e| u16::try_from(e).vortex_expect("end value fits in u16"))
                .collect();
            PrimitiveArray::new(Buffer::copy_from(ends_typed), Validity::NonNullable).into_array()
        }
        PType::U32 => {
            let ends_typed: Vec<u32> = ends
                .iter()
                .map(|&e| u32::try_from(e).vortex_expect("end value fits in u32"))
                .collect();
            PrimitiveArray::new(Buffer::copy_from(ends_typed), Validity::NonNullable).into_array()
        }
        PType::U64 => {
            PrimitiveArray::new(Buffer::copy_from(ends), Validity::NonNullable).into_array()
        }
        _ => unreachable!("Only unsigned integer types are valid for ends"),
    };

    Ok(ends_array)
}
