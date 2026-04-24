// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::PrimitiveArray;
use crate::dtype::NativePType;
use crate::dtype::half::f16;
use crate::match_each_native_ptype;

cfg_if::cfg_if! {
    if #[cfg(target_feature = "avx2")] {
        pub const IS_CONST_LANE_WIDTH: usize = 32;
    } else {
        pub const IS_CONST_LANE_WIDTH: usize = 16;
    }
}

/// Assumes any floating point has been cast into its bit representation for which != and !is_eq are the same
/// Assumes there's at least 1 value in the slice, which is an invariant of the entry level function.
pub fn compute_is_constant<T: NativePType, const WIDTH: usize>(values: &[T]) -> bool {
    let first_value = values[0];
    let first_vec = &[first_value; WIDTH];

    let mut chunks = values[1..].chunks_exact(WIDTH);
    for chunk in &mut chunks {
        assert_eq!(chunk.len(), WIDTH); // let the compiler know each chunk is WIDTH.
        if first_vec != chunk {
            return false;
        }
    }

    for value in chunks.remainder() {
        if !value.is_eq(first_value) {
            return false;
        }
    }

    true
}

trait EqFloat {
    type IntType;
}

impl EqFloat for f16 {
    type IntType = u16;
}
impl EqFloat for f32 {
    type IntType = u32;
}
impl EqFloat for f64 {
    type IntType = u64;
}

pub(super) fn check_primitive_constant(array: &PrimitiveArray) -> bool {
    match_each_native_ptype!(array.ptype(), integral: |P| {
        compute_is_constant::<_, {IS_CONST_LANE_WIDTH / size_of::<P>()}>(array.as_slice::<P>())
    }, floating: |P| {
        compute_is_constant::<_, {IS_CONST_LANE_WIDTH / size_of::<P>()}>(unsafe { std::mem::transmute::<&[P], &[<P as EqFloat>::IntType]>(array.as_slice::<P>()) })
    })
}
