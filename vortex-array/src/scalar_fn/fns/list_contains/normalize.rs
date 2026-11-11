// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Float normalization, which lets a bitwise membership lookup give the equality of a
//! [`CompareMode`].
//!
//! Membership needs only equality. In each compare mode, two floats are equal exactly when their
//! normalized bits are equal:
//!
//! - With [`SignedZeros::Equal`], `-0.0` becomes `+0.0`.
//! - With [`NanOrder::Canonical`], each `NaN` becomes one canonical `NaN`.
//! - With [`NanOrder::Ieee754`], a `NaN` of the set becomes one `NaN`, and a `NaN` of a needle
//!   becomes a `NaN` with other bits. So a `NaN` needle matches no element, as in IEEE 754.
//!
//! The normalized values only feed a lookup. They are never part of a result.

use num_traits::Float;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::ListViewArray;
use crate::arrays::MapArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::StructArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::map::MapArraySlotsExt;
use crate::arrays::struct_::StructArrayExt;
use crate::dtype::DType;
use crate::match_each_float_ptype;
use crate::scalar_fn::fns::operators::CompareMode;
use crate::scalar_fn::fns::operators::NanOrder;
use crate::scalar_fn::fns::operators::SignedZeros;
use crate::scalar_fn::fns::operators::contains_float;

/// How to normalize the floats of a membership lookup. The default is no normalization, which
/// gives bitwise equality.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct FloatNormalization {
    zeros_equal: bool,
    nan_order: NanOrder,
}

/// The side of a membership lookup that a value is on.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(super) enum Side {
    /// An element of the list.
    Element,
    /// A needle that is looked up in the list.
    Needle,
}

impl FloatNormalization {
    /// The number of distinct normalizations, which [`Self::index`] numbers.
    pub(super) const COUNT: usize = 6;

    /// The normalization that gives the equality of `mode` for values of `dtype`.
    pub(super) fn new(mode: CompareMode, dtype: &DType) -> Self {
        if !contains_float(dtype) {
            return Self::default();
        }

        Self {
            zeros_equal: mode.signed_zeros() == SignedZeros::Equal,
            nan_order: mode.nan_order(),
        }
    }

    /// Whether this normalization leaves each value unchanged.
    pub(super) fn is_identity(self) -> bool {
        self == Self::default()
    }

    /// A number below [`Self::COUNT`] that identifies this normalization. The identity is `0`.
    pub(super) fn index(self) -> usize {
        let nan = match self.nan_order {
            NanOrder::TotalOrder => 0,
            NanOrder::Canonical => 1,
            NanOrder::Ieee754 => 2,
        };
        usize::from(self.zeros_equal) * 3 + nan
    }

    /// Normalizes the floats of `array`, at the top level and in nested values, for `side`.
    pub(super) fn apply(
        self,
        array: &ArrayRef,
        side: Side,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        if self.is_identity() || !contains_float(array.dtype()) {
            return Ok(array.clone());
        }

        let validity = array.validity()?;

        Ok(match array.dtype() {
            DType::Primitive(ptype, _) => {
                let primitive = array.clone().execute::<PrimitiveArray>(ctx)?;
                match_each_float_ptype!(*ptype, |T| {
                    let values = primitive
                        .as_slice::<T>()
                        .iter()
                        .map(|&value| self.normalize(value, side))
                        .collect::<Vec<T>>();
                    PrimitiveArray::new(values, validity).into_array()
                })
            }
            DType::Extension(ext_dtype) => {
                let extension = array.clone().execute::<ExtensionArray>(ctx)?;
                let storage = self.apply(extension.storage_array(), side, ctx)?;
                ExtensionArray::try_new(ext_dtype.clone(), storage)?.into_array()
            }
            DType::Struct(..) => {
                let struct_ = array.clone().execute::<StructArray>(ctx)?;
                let fields = struct_
                    .iter_unmasked_fields()
                    .map(|field| self.apply(field, side, ctx))
                    .collect::<VortexResult<Vec<_>>>()?;
                StructArray::try_new(struct_.names().clone(), fields, array.len(), validity)?
                    .into_array()
            }
            DType::List(..) => {
                let list = array.clone().execute::<ListViewArray>(ctx)?;
                let elements = self.apply(list.elements(), side, ctx)?;
                ListViewArray::try_new(
                    elements,
                    list.offsets().clone(),
                    list.sizes().clone(),
                    validity,
                    ctx,
                )?
                .into_array()
            }
            DType::FixedSizeList(_, size, _) => {
                let list = array.clone().execute::<FixedSizeListArray>(ctx)?;
                let elements = self.apply(list.elements(), side, ctx)?;
                FixedSizeListArray::try_new(elements, *size, validity, array.len())?.into_array()
            }
            DType::Map(map_dtype, _) => {
                let map = array.clone().execute::<MapArray>(ctx)?;
                let entries = self
                    .apply(map.entries(), side, ctx)?
                    .execute::<ListViewArray>(ctx)?;
                MapArray::try_new(map_dtype.clone(), entries)?.into_array()
            }
            DType::Null
            | DType::Bool(_)
            | DType::Decimal(..)
            | DType::Utf8(_)
            | DType::Binary(_)
            | DType::Union(..)
            | DType::Variant(_) => array.clone(),
        })
    }

    /// Normalizes one float for `side`.
    fn normalize<T: Float>(self, value: T, side: Side) -> T {
        if value.is_nan() {
            return match (self.nan_order, side) {
                (NanOrder::TotalOrder, _) => value,
                (NanOrder::Canonical, _) | (NanOrder::Ieee754, Side::Element) => T::nan(),
                (NanOrder::Ieee754, Side::Needle) => -T::nan(),
            };
        }

        if self.zeros_equal && value.is_zero() {
            return T::zero();
        }
        value
    }
}
