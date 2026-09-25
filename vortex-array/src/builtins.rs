// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A collection of built-in common scalar functions.
//!
//! It is expected that each Vortex integration may provide its own set of scalar functions with
//! semantics that exactly match the underlying system (e.g. SQL engine, DataFrame library, etc).
//!
//! This set of functions should cover the basics, and in general leans towards the semantics of
//! the equivalent Arrow compute function.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::InterleaveArray;
use crate::dtype::DType;
use crate::dtype::FieldName;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::between::Between;
use crate::scalar_fn::fns::between::BetweenOptions;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::fill_null::FillNull;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::is_not_null::IsNotNull;
use crate::scalar_fn::fns::is_null::IsNull;
use crate::scalar_fn::fns::list_contains::ListContains;
use crate::scalar_fn::fns::mask::Mask;
use crate::scalar_fn::fns::not::Not;
use crate::scalar_fn::fns::operators::Operator;
use crate::scalar_fn::fns::zip::Zip;

pub trait ArrayBuiltins: Sized {
    /// Cast to the given data type.
    fn cast(&self, dtype: DType) -> VortexResult<ArrayRef>;

    /// Replace null values with the given fill value.
    fn fill_null(&self, fill_value: impl Into<Scalar>) -> VortexResult<ArrayRef>;

    /// Get item by field name (for struct types).
    fn get_item(&self, field_name: impl Into<FieldName>) -> VortexResult<ArrayRef>;

    /// Is null check.
    fn is_null(&self) -> VortexResult<ArrayRef>;

    /// Is not null check.
    fn is_not_null(&self) -> VortexResult<ArrayRef>;

    /// Mask the array using the given boolean mask.
    /// The resulting array's validity is the intersection of the original array's validity
    /// and the mask's validity.
    fn mask(self, mask: ArrayRef) -> VortexResult<ArrayRef>;

    /// Boolean negation.
    fn not(&self) -> VortexResult<ArrayRef>;

    /// Conditional selection: `result[i] = if mask[i] then if_true[i] else if_false[i]`.
    fn zip(&self, if_true: ArrayRef, if_false: ArrayRef) -> VortexResult<ArrayRef>;

    /// Random-access gather by `(array_index, row_index)`: output row `i` is taken from
    /// `values[array_indices[i]][row_indices[i]]`, where `self` is the (non-nullable)
    /// `array_indices` selector and `row_indices` names the position within the selected value.
    /// See [`InterleaveArray`].
    fn interleave(
        &self,
        values: impl IntoIterator<Item = ArrayRef>,
        row_indices: ArrayRef,
    ) -> VortexResult<ArrayRef>;

    /// Check if a list contains a value.
    fn list_contains(&self, value: ArrayRef) -> VortexResult<ArrayRef>;

    /// Apply a binary operator to this array and another.
    fn binary(&self, rhs: ArrayRef, op: Operator) -> VortexResult<ArrayRef>;

    /// Compare a values between lower </<= value </<= upper
    fn between(
        self,
        lower: ArrayRef,
        upper: ArrayRef,
        options: BetweenOptions,
    ) -> VortexResult<ArrayRef>;
}

impl ArrayBuiltins for ArrayRef {
    fn cast(&self, dtype: DType) -> VortexResult<ArrayRef> {
        if self.dtype() == &dtype {
            return Ok(self.clone());
        }
        Cast::new(self.clone(), dtype).into_array().optimize()
    }

    fn fill_null(&self, fill_value: impl Into<Scalar>) -> VortexResult<ArrayRef> {
        let fill_value = fill_value.into();
        if !self.dtype().is_nullable() {
            return self.cast(fill_value.dtype().clone());
        }
        FillNull::try_new(
            self.clone(),
            ConstantArray::new(fill_value, self.len()).into_array(),
        )?
        .into_array()
        .optimize()
    }

    fn get_item(&self, field_name: impl Into<FieldName>) -> VortexResult<ArrayRef> {
        GetItem::try_new(self.clone(), field_name)?
            .into_array()
            .optimize()
    }

    fn is_null(&self) -> VortexResult<ArrayRef> {
        IsNull::new(self.clone()).into_array().optimize()
    }

    fn is_not_null(&self) -> VortexResult<ArrayRef> {
        IsNotNull::new(self.clone()).into_array().optimize()
    }

    fn mask(self, mask: ArrayRef) -> VortexResult<ArrayRef> {
        Mask::try_new(self, mask)?.into_array().optimize()
    }

    fn not(&self) -> VortexResult<ArrayRef> {
        Not::try_new(self.clone())?.into_array().optimize()
    }

    fn zip(&self, if_true: ArrayRef, if_false: ArrayRef) -> VortexResult<ArrayRef> {
        Ok(Zip::try_new(if_true, if_false, self.clone())?.into_array())
    }

    fn interleave(
        &self,
        values: impl IntoIterator<Item = ArrayRef>,
        row_indices: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        Ok(
            InterleaveArray::try_new(values.into_iter().collect(), self.clone(), row_indices)?
                .into_array(),
        )
    }

    fn list_contains(&self, value: ArrayRef) -> VortexResult<ArrayRef> {
        ListContains::try_new(self.clone(), value)?
            .into_array()
            .optimize()
    }

    fn binary(&self, rhs: ArrayRef, op: Operator) -> VortexResult<ArrayRef> {
        Binary::try_new(self.clone(), rhs, op)?
            .into_array()
            .optimize()
    }

    fn between(
        self,
        lower: ArrayRef,
        upper: ArrayRef,
        options: BetweenOptions,
    ) -> VortexResult<ArrayRef> {
        Between::try_new(self, lower, upper, options)?
            .into_array()
            .optimize()
    }
}
