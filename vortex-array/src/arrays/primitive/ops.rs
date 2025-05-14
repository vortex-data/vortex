use vortex_dtype::half::f16;
use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::arrays::{PrimitiveArray, PrimitiveVTable};
use crate::vtable::{OperationsVTable, ValidityHelper};
use crate::{ArrayRef, Cost, IntoArray};

cfg_if::cfg_if! {
    if #[cfg(target_feature = "avx2")] {
        pub const IS_CONST_LANE_WIDTH: usize = 32;
    } else {
        pub const IS_CONST_LANE_WIDTH: usize = 16;
    }
}

impl OperationsVTable<PrimitiveVTable> for PrimitiveVTable {
    fn slice(array: &PrimitiveArray, start: usize, stop: usize) -> VortexResult<ArrayRef> {
        match_each_native_ptype!(array.ptype(), |$T| {
            Ok(PrimitiveArray::new(
                array.buffer::<$T>().slice(start..stop),
                array.validity().slice(start, stop)?,
            )
            .into_array())
        })
    }

    fn scalar_at(array: &PrimitiveArray, index: usize) -> VortexResult<Scalar> {
        Ok(match_each_native_ptype!(array.ptype(), |$T| {
            Scalar::primitive(array.as_slice::<$T>()[index], array.dtype().nullability())
        }))
    }

    fn is_constant(array: &PrimitiveArray, cost: Cost) -> VortexResult<Option<bool>> {
        if cost.is_negligible() {
            return Ok(None);
        }
        let is_constant = match_each_native_ptype!(array.ptype(), integral: |$P| {
            compute_is_constant::<_, {IS_CONST_LANE_WIDTH / size_of::<$P>()}>(array.as_slice::<$P>())
        } floating_point: |$P| {
            compute_is_constant::<_, {IS_CONST_LANE_WIDTH / size_of::<$P>()}>(unsafe { std::mem::transmute::<&[$P], &[<$P as EqFloat>::IntType]>(array.as_slice::<$P>()) })
        });

        Ok(Some(is_constant))
    }
}

// Assumes any floating point has been cast into its bit representation for which != and !is_eq are the same
// Assumes there's at least 1 value in the slice, which is an invariant of the entry level function.
pub fn compute_is_constant<T: NativePType, const WIDTH: usize>(values: &[T]) -> bool {
    let first_value = values[0];
    let first_vec = &[first_value; WIDTH];

    let mut chunks = values[1..].array_chunks::<WIDTH>();
    for chunk in &mut chunks {
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
