// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arbitrary::Arbitrary;
use arbitrary::Result;
use arbitrary::Unstructured;
use arbitrary::unstructured::Int;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_array::arrays::arbitrary::ArbitraryArrayConfig;
use vortex_array::arrays::arbitrary::ArbitraryWith;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::UnsignedPType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;

use crate::RunEnd;
use crate::RunEndArray;

/// A wrapper type to implement `Arbitrary` for `RunEndArray`.
#[derive(Clone, Debug)]
pub struct ArbitraryRunEndArray(pub RunEndArray);

impl<'a> Arbitrary<'a> for ArbitraryRunEndArray {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        // Pick a random primitive type for values.
        let ptype: PType = u.arbitrary()?;
        let nullability: Nullability = u.arbitrary()?;
        let dtype = DType::Primitive(ptype, nullability);
        Self::with_dtype(u, &dtype)
    }
}

impl ArbitraryRunEndArray {
    /// Generate an arbitrary RunEndArray with the given dtype for values.
    ///
    /// The dtype must be a primitive or boolean type.
    pub fn with_dtype(u: &mut Unstructured, dtype: &DType) -> Result<Self> {
        // Number of runs (values/ends pairs)
        let num_runs = u.int_in_range(0..=20)?;

        if num_runs == 0 {
            // Empty RunEndArray
            let ends = unsafe {
                PrimitiveArray::new_unchecked(Buffer::<u32>::empty(), Validity::NonNullable)
                    .into_array()
            };
            let values = ArbitraryArray::arbitrary_with_config(
                u,
                &ArbitraryArrayConfig {
                    dtype: Some(dtype.clone()),
                    len: 0..=0,
                },
            )?
            .0;
            let runend_array = unsafe { RunEnd::new_unchecked(ends, values, 0, 0) };
            return Ok(ArbitraryRunEndArray(runend_array));
        }

        // Generate arbitrary values for each run
        let values = ArbitraryArray::arbitrary_with_config(
            u,
            &ArbitraryArrayConfig {
                dtype: Some(dtype.clone()),
                len: num_runs..=num_runs,
            },
        )?
        .0;

        let len = u.int_in_range(0..=2048)?;
        // Generate strictly increasing ends
        // Each end must be > previous end, and first end must be >= 1
        let ends = random_strictly_sorted_ends(u, num_runs, len)?;

        let runend_array = unsafe { RunEnd::new_unchecked(ends, values, 0, len) };

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
    target_len: usize,
) -> Result<ArrayRef> {
    // Choose a random unsigned PType for ends
    let mut ends_ptypes = vec![PType::U8, PType::U16, PType::U32, PType::U64];
    if target_len >= u8::MAX as usize {
        ends_ptypes.remove(0);
    }
    if target_len >= u16::MAX as usize {
        ends_ptypes.remove(0);
    }
    if target_len >= u32::MAX as usize {
        ends_ptypes.remove(0);
    }
    let ends_ptype = *u.choose(&ends_ptypes)?;

    match ends_ptype {
        PType::U8 => random_strictly_sorted(
            u,
            num_runs,
            u8::try_from(target_len).vortex_expect("must fit in u8"),
        ),
        PType::U16 => random_strictly_sorted(
            u,
            num_runs,
            u16::try_from(target_len).vortex_expect("must fit in u16"),
        ),
        PType::U32 => random_strictly_sorted(
            u,
            num_runs,
            u32::try_from(target_len).vortex_expect("must fit in u32"),
        ),
        PType::U64 => random_strictly_sorted(
            u,
            num_runs,
            u64::try_from(target_len).vortex_expect("must fit in u64"),
        ),
        _ => unreachable!("Only unsigned integer types are valid for ends"),
    }
}

fn random_strictly_sorted<T: UnsignedPType + Int>(
    u: &mut Unstructured,
    num_runs: usize,
    target: T,
) -> Result<ArrayRef> {
    // Generate strictly increasing values
    // Start from 0, increment by at least 1 each time
    let mut ends: Vec<T> = Vec::with_capacity(num_runs);
    let mut current = T::zero();

    for i in 0..num_runs {
        // Each run must have at least length 1, so increment by at least 1
        let increment = match i == num_runs - 1 {
            true => {
                // Last element should reach target_len
                if target > current {
                    target - current
                } else {
                    T::one()
                }
            }
            false => {
                // Random increment between 1 and 10
                u.int_in_range(T::one()..=T::from(10).vortex_expect("10 will fit in all T"))?
            }
        };
        current += increment;
        ends.push(current);
    }

    Ok(PrimitiveArray::new(Buffer::copy_from(ends), Validity::NonNullable).into_array())
}
