// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared input decoding and Vortex output construction for `geo_types` kernels.
//!
//! `geo_types` is the row representation consumed by the kernel. These helpers always construct
//! and return Vortex arrays; they do not expose `geo_types` values as scalar-function outputs.

use geo_types::Geometry;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::extension::geometries;

/// A primitive result produced after kernel inputs are decoded to `geo_types`.
pub(crate) trait GeoTypesOutput: Copy {
    /// The Vortex dtype used to represent this output.
    fn dtype(nullability: Nullability) -> DType;

    /// Convert one computed value into a Vortex scalar for constant output.
    fn into_scalar(self, nullability: Nullability) -> Scalar;

    /// Scatter values computed for valid rows into a full-length output array.
    fn build_array(
        len: usize,
        valid: &Mask,
        values: Vec<Self>,
        nullability: Nullability,
    ) -> ArrayRef;
}

impl GeoTypesOutput for f64 {
    fn dtype(nullability: Nullability) -> DType {
        DType::Primitive(PType::F64, nullability)
    }

    fn into_scalar(self, nullability: Nullability) -> Scalar {
        Scalar::primitive(self, nullability)
    }

    fn build_array(
        len: usize,
        valid: &Mask,
        values: Vec<Self>,
        nullability: Nullability,
    ) -> ArrayRef {
        let validity = Validity::from_mask(valid.clone(), nullability);
        match valid.indices() {
            AllOr::All => PrimitiveArray::new(values, validity).into_array(),
            AllOr::None => PrimitiveArray::new(vec![0.0f64; len], validity).into_array(),
            AllOr::Some(rows) => {
                let mut data = vec![0.0f64; len];
                for (&row, value) in rows.iter().zip(values) {
                    data[row] = value;
                }
                PrimitiveArray::new(data, validity).into_array()
            }
        }
    }
}

impl GeoTypesOutput for bool {
    fn dtype(nullability: Nullability) -> DType {
        DType::Bool(nullability)
    }

    fn into_scalar(self, nullability: Nullability) -> Scalar {
        Scalar::bool(self, nullability)
    }

    fn build_array(
        len: usize,
        valid: &Mask,
        values: Vec<Self>,
        nullability: Nullability,
    ) -> ArrayRef {
        let validity = Validity::from_mask(valid.clone(), nullability);
        match valid.indices() {
            AllOr::All => BoolArray::new(BitBuffer::from_iter(values), validity).into_array(),
            AllOr::None => BoolArray::new(BitBuffer::new_unset(len), validity).into_array(),
            AllOr::Some(rows) => {
                let mut data = vec![false; len];
                for (&row, value) in rows.iter().zip(values) {
                    data[row] = value;
                }
                BoolArray::new(BitBuffer::from_iter(data), validity).into_array()
            }
        }
    }
}

/// Evaluate a decoded kernel over each valid row of one geometry column.
pub(super) fn eval_column<T, F>(
    column: &ArrayRef,
    valid: &Mask,
    compute: F,
    nullability: Nullability,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: GeoTypesOutput,
    F: Fn(&Geometry<f64>) -> T,
{
    let len = column.len();
    let decoded = geometries(&column.filter(valid.clone())?, ctx)?;
    let values = decoded.iter().map(compute).collect();
    Ok(T::build_array(len, valid, values, nullability))
}
