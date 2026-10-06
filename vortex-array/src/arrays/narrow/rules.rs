// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Push selection and compatible scalar operations into Narrow values.
//!
//! Each reduction preserves the logical integer width and derives result nullability from the
//! transformed child. Constants and other Narrow children can require a wider stored dtype.

use std::ops::Range;

use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::Narrow;
use super::NarrowArray;
use super::NarrowArraySlotsExt;
use super::merge::ChunkedInputsReduce;
use super::merge::InterleaveInputsReduce;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::ArrayView;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::Dict;
use crate::arrays::DictArray;
use crate::arrays::ScalarFn;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::dict::DictSlots;
use crate::arrays::dict::TakeReduce;
use crate::arrays::dict::TakeReduceAdaptor;
use crate::arrays::filter::FilterReduce;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::scalar_fn::ExactScalarFn;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::arrays::scalar_fn::ScalarFnArrayView;
use crate::arrays::slice::SliceReduce;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::between::BetweenReduce;
use crate::scalar_fn::fns::between::BetweenReduceAdaptor;
use crate::scalar_fn::fns::cast::CastReduce;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::fill_null::FillNullReduce;
use crate::scalar_fn::fns::fill_null::FillNullReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduce;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;
use crate::scalar_fn::fns::operators::Operator;
use crate::scalar_fn::fns::zip::Zip;
use crate::scalar_fn::fns::zip::ZipReduce;
use crate::scalar_fn::fns::zip::ZipReduceAdaptor;

pub(super) const RULES: ParentRuleSet<Narrow> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&BetweenReduceAdaptor(Narrow)),
    ParentRuleSet::lift(&CastReduceAdaptor(Narrow)),
    ParentRuleSet::lift(&ChunkedInputsReduce),
    ParentRuleSet::lift(&FillNullReduceAdaptor(Narrow)),
    ParentRuleSet::lift(&FilterReduceAdaptor(Narrow)),
    ParentRuleSet::lift(&InterleaveInputsReduce),
    ParentRuleSet::lift(&MaskReduceAdaptor(Narrow)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Narrow)),
    ParentRuleSet::lift(&TakeReduceAdaptor(Narrow)),
    ParentRuleSet::lift(&TakeIndicesReduce),
    ParentRuleSet::lift(&ZipReduceAdaptor(Narrow)),
    ParentRuleSet::lift(&ZipFalseReduce),
]);

pub(super) fn rewrap(
    array: ArrayView<'_, Narrow>,
    values: ArrayRef,
) -> VortexResult<Option<ArrayRef>> {
    let dtype = array.dtype().with_nullability(values.dtype().nullability());
    if values.dtype() == &dtype {
        return Ok(Some(values));
    }
    Ok(Some(NarrowArray::try_new(values, dtype)?.into_array()))
}

impl SliceReduce for Narrow {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        rewrap(array, array.values().slice(range)?)
    }
}

impl FilterReduce for Narrow {
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        rewrap(array, array.values().filter(mask.clone())?)
    }
}

impl TakeReduce for Narrow {
    fn take(array: ArrayView<'_, Self>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        rewrap(array, array.values().take(indices.clone())?)
    }
}

impl MaskReduce for Narrow {
    fn mask(array: ArrayView<'_, Self>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        rewrap(array, array.values().clone().mask(mask.clone())?)
    }
}

impl CastReduce for Narrow {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let (Ok(target), Ok(storage)) = (
            PType::try_from(dtype),
            PType::try_from(array.values().dtype()),
        ) else {
            return Ok(None);
        };
        if target.is_float() {
            return array.values().cast(dtype.clone()).map(Some);
        }
        // A same-width unsigned value may exceed the signed range. The next signed width can
        // represent every source value, while narrowing casts still use the checked cast kernel.
        let storage_type = if storage.is_unsigned_int() && target.is_signed_int() {
            [PType::I16, PType::I32, PType::I64]
                .into_iter()
                .find(|ptype| ptype.byte_width() > storage.byte_width())
                .unwrap_or(target)
        } else if target.is_signed_int() {
            storage.to_signed()
        } else {
            storage.to_unsigned()
        };
        if target.byte_width() <= storage_type.byte_width() {
            return array.values().cast(dtype.clone()).map(Some);
        }
        let storage_dtype = DType::Primitive(storage_type, dtype.nullability());
        Ok(Some(
            NarrowArray::try_new(array.values().cast(storage_dtype)?, dtype.clone())?.into_array(),
        ))
    }
}

impl FillNullReduce for Narrow {
    fn fill_null(
        array: ArrayView<'_, Self>,
        fill_value: &Scalar,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(ptype) = scalar_storage_type(array, fill_value)? else {
            return Ok(None);
        };
        let storage_dtype = DType::Primitive(ptype, array.dtype().nullability());
        let fill_dtype = DType::Primitive(ptype, fill_value.dtype().nullability());
        rewrap(
            array,
            array
                .values()
                .cast(storage_dtype)?
                .fill_null(fill_value.cast(&fill_dtype)?)?,
        )
    }
}

/// Select a common primitive width without reading the child's values.
pub(super) fn scalar_storage_type(
    array: ArrayView<'_, Narrow>,
    scalar: &Scalar,
) -> VortexResult<Option<PType>> {
    let (Ok(storage), Ok(logical)) = (
        PType::try_from(array.values().dtype()),
        PType::try_from(array.dtype()),
    ) else {
        return Ok(None);
    };
    let required = if scalar.is_null() {
        storage
    } else if logical.is_signed_int() {
        PType::min_signed_ptype_for_value(i64::try_from(scalar)?)
    } else {
        PType::min_unsigned_ptype_for_value(u64::try_from(scalar)?)
    };
    let ptype = if storage.byte_width() >= required.byte_width() {
        storage
    } else {
        required
    };
    Ok((ptype.byte_width() <= logical.byte_width()).then_some(ptype))
}

impl ZipReduce for Narrow {
    fn zip(
        array: ArrayView<'_, Self>,
        if_false: &ArrayRef,
        mask: &ArrayRef,
    ) -> VortexResult<Option<ArrayRef>> {
        zip_values(array, if_false, mask, true)
    }
}

#[derive(Debug)]
struct ZipFalseReduce;

impl ArrayParentReduceRule<Narrow> for ZipFalseReduce {
    type Parent = ExactScalarFn<Zip>;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Narrow>,
        parent: ScalarFnArrayView<'_, Zip>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 1 {
            return Ok(None);
        }
        let Some(parent) = parent.as_opt::<ScalarFn>() else {
            return Ok(None);
        };
        zip_values(array, parent.get_child(0), parent.get_child(2), false)
    }
}

fn zip_values(
    array: ArrayView<'_, Narrow>,
    other: &ArrayRef,
    mask: &ArrayRef,
    is_true: bool,
) -> VortexResult<Option<ArrayRef>> {
    let (values, ptype) = if let Some(other) = other.as_opt::<Narrow>() {
        let (Ok(lhs), Ok(rhs)) = (
            PType::try_from(array.values().dtype()),
            PType::try_from(other.values().dtype()),
        ) else {
            return Ok(None);
        };
        let ptype = if lhs.byte_width() >= rhs.byte_width() {
            lhs
        } else {
            rhs
        };
        (other.values().clone(), ptype)
    } else if let Some(scalar) = other.as_constant() {
        let Some(ptype) = scalar_storage_type(array, &scalar)? else {
            return Ok(None);
        };
        (other.clone(), ptype)
    } else {
        return Ok(None);
    };
    let lhs_dtype = DType::Primitive(ptype, array.dtype().nullability());
    let rhs_dtype = DType::Primitive(ptype, other.dtype().nullability());
    let array_values = array.values().cast(lhs_dtype)?;
    let other_values = values.cast(rhs_dtype)?;
    let values = if is_true {
        mask.zip(array_values, other_values)?
    } else {
        mask.zip(other_values, array_values)?
    };
    rewrap(array, values)
}

#[derive(Debug)]
struct TakeIndicesReduce;

impl ArrayParentReduceRule<Narrow> for TakeIndicesReduce {
    type Parent = Dict;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Narrow>,
        parent: ArrayView<'_, Dict>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx != 0 {
            return Ok(None);
        }
        // Dictionary codes are internal indices. Their integer width does not determine the
        // result dtype, so the dictionary can consume the stored child directly.
        Ok(Some(
            DictArray::try_from_parts(
                ArrayParts::new(
                    Dict,
                    parent.dtype().clone(),
                    parent.len(),
                    parent.data().clone(),
                )
                .with_slots(
                    DictSlots {
                        codes: array.values().clone(),
                        values: parent.values().clone(),
                    }
                    .into_slots(),
                ),
            )?
            .into_array(),
        ))
    }
}

impl BetweenReduce for Narrow {
    fn between(
        array: ArrayView<'_, Self>,
        lower: &ArrayRef,
        upper: &ArrayRef,
        options: &BetweenOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        let (Some(lower_value), Some(upper_value)) = (lower.as_constant(), upper.as_constant())
        else {
            // Binary comparison already handles children with different stored widths.
            return two_compares(array, lower, upper, options).map(Some);
        };
        let storage_dtype = array.values().dtype();
        if let (Ok(lower), Ok(upper)) = (
            lower_value.cast(&storage_dtype.with_nullability(lower.dtype().nullability())),
            upper_value.cast(&storage_dtype.with_nullability(upper.dtype().nullability())),
        ) {
            return array
                .values()
                .clone()
                .between(
                    ConstantArray::new(lower, array.len()).into_array(),
                    ConstantArray::new(upper, array.len()).into_array(),
                    options.clone(),
                )
                .map(Some);
        }
        two_compares(array, lower, upper, options).map(Some)
    }
}

fn two_compares(
    array: ArrayView<'_, Narrow>,
    lower: &ArrayRef,
    upper: &ArrayRef,
    options: &BetweenOptions,
) -> VortexResult<ArrayRef> {
    let lower_op = if options.lower_strict.is_strict() {
        Operator::Gt
    } else {
        Operator::Gte
    };
    let upper_op = if options.upper_strict.is_strict() {
        Operator::Lt
    } else {
        Operator::Lte
    };
    let lhs = array.as_ref().binary(lower.clone(), lower_op)?;
    let rhs = array.as_ref().binary(upper.clone(), upper_op)?;
    lhs.binary(rhs, Operator::And)
}
