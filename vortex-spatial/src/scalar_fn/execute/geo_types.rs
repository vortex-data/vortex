// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared input decoding and Vortex output construction for `geo_types` kernels.
//!
//! `geo_types` is the row representation consumed by the kernel. These helpers always construct
//! and return Vortex arrays; they do not expose `geo_types` values as scalar-function outputs.

use geo_types::Geometry;
use geo_types::MultiPolygon as GeoMultiPolygon;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::extension::build_multipolygon_array;
use crate::extension::geometries;

/// A Vortex representation for values produced by a `geo_types` kernel.
pub(crate) trait GeoTypesOutput: Sized {
    /// Build an array from values computed for rows selected by `valid`.
    fn build_array(
        len: usize,
        valid: &Mask,
        values: Vec<Self>,
        output_dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef>;

    /// Build a repeated result for two constant operands.
    fn build_constant(
        value: Self,
        len: usize,
        output_dtype: &DType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let one = Self::build_array(1, &Mask::new_true(1), vec![value], output_dtype, ctx)?;
        Ok(ConstantArray::new(one.execute_scalar(0, ctx)?, len).into_array())
    }
}

/// Scatter values computed for valid rows into a full-length nullable vector.
pub(crate) fn scatter_valid<T>(len: usize, valid: &Mask, values: Vec<T>) -> Vec<Option<T>> {
    match valid.indices() {
        AllOr::All => values.into_iter().map(Some).collect(),
        AllOr::None => (0..len).map(|_| None).collect(),
        AllOr::Some(rows) => {
            let mut output = (0..len).map(|_| None).collect::<Vec<Option<T>>>();
            for (&row, value) in rows.iter().zip(values) {
                output[row] = Some(value);
            }
            output
        }
    }
}

impl GeoTypesOutput for f64 {
    fn build_array(
        len: usize,
        valid: &Mask,
        values: Vec<Self>,
        output_dtype: &DType,
        _: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let validity = Validity::from_mask(valid.clone(), output_dtype.nullability());
        Ok(match valid.indices() {
            AllOr::All => PrimitiveArray::new(values, validity).into_array(),
            AllOr::None => PrimitiveArray::new(vec![0.0; len], validity).into_array(),
            AllOr::Some(rows) => {
                let mut data = vec![0.0; len];
                for (&row, value) in rows.iter().zip(values) {
                    data[row] = value;
                }
                PrimitiveArray::new(data, validity).into_array()
            }
        })
    }
}

impl GeoTypesOutput for bool {
    fn build_array(
        len: usize,
        valid: &Mask,
        values: Vec<Self>,
        output_dtype: &DType,
        _: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let validity = Validity::from_mask(valid.clone(), output_dtype.nullability());
        Ok(match valid.indices() {
            AllOr::All => BoolArray::new(BitBuffer::from_iter(values), validity).into_array(),
            AllOr::None => BoolArray::new(BitBuffer::new_unset(len), validity).into_array(),
            AllOr::Some(rows) => {
                let mut data = vec![false; len];
                for (&row, value) in rows.iter().zip(values) {
                    data[row] = value;
                }
                BoolArray::new(BitBuffer::from_iter(data), validity).into_array()
            }
        })
    }
}

impl GeoTypesOutput for GeoMultiPolygon<f64> {
    fn build_array(
        len: usize,
        valid: &Mask,
        values: Vec<Self>,
        output_dtype: &DType,
        _: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let output_dtype = output_dtype.as_extension();
        build_multipolygon_array(&scatter_valid(len, valid, values), output_dtype)
    }
}

/// Evaluate a decoded kernel over each valid row of one geometry column.
pub(super) fn eval_column<T, F>(
    column: &ArrayRef,
    valid: &Mask,
    compute: F,
    output_dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef>
where
    T: GeoTypesOutput,
    F: Fn(&Geometry<f64>) -> T,
{
    let len = column.len();
    let decoded = geometries(&column.filter(valid.clone())?, ctx)?;
    let values = decoded.iter().map(compute).collect();
    T::build_array(len, valid, values, output_dtype, ctx)
}
