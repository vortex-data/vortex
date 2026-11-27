// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Float normalization, which lets a bitwise membership lookup give the equality of a
//! [`CompareMode`].
//!
//! Membership needs only equality. In each compare mode, two floats are equal exactly when their
//! normalized bits are equal:
//!
//! - With each order but [`FloatOrder::TotalOrder`], `-0.0` becomes `+0.0`.
//! - With [`FloatOrder::Canonical`], each `NaN` becomes one canonical `NaN`.
//! - With [`FloatOrder::Ieee754`], a `NaN` of the set becomes one `NaN`, and a `NaN` of a needle
//!   becomes a `NaN` with other bits. So a `NaN` needle matches no element, as in IEEE 754.
//!
//! The normalized values only feed a lookup. They are never part of a result.

use num_traits::Float;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::map::MapArraySlotsExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::match_each_float_ptype;
use vortex_array::scalar_fn::builtin::operators::CompareMode;
use vortex_array::scalar_fn::builtin::operators::FloatOrder;
use vortex_array::scalar_fn::builtin::operators::contains_float;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

/// How to normalize the floats of a membership lookup, so that bitwise equality of the normalized
/// values is the equality of a [`FloatOrder`].
///
/// [`FloatOrder`] is an open enum. This enum holds only the orders that have a normalization, so
/// its matches are complete.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub(super) enum FloatNormalization {
    /// Leaves each value unchanged, for [`FloatOrder::TotalOrder`] and for values without floats.
    #[default]
    Identity,
    /// The normalization of [`FloatOrder::ZerosEqual`].
    ZerosEqual,
    /// The normalization of [`FloatOrder::Canonical`].
    Canonical,
    /// The normalization of [`FloatOrder::Ieee754`].
    Ieee754,
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
    pub(super) const COUNT: usize = 4;

    /// The normalization that gives the equality of `mode` for values of `dtype`.
    ///
    /// # Errors
    ///
    /// Returns an error if the order of floats in `mode` has no normalization.
    pub(super) fn try_new(mode: CompareMode, dtype: &DType) -> VortexResult<Self> {
        if !contains_float(dtype) {
            return Ok(Self::Identity);
        }

        Ok(match mode.float_order() {
            FloatOrder::TotalOrder => Self::Identity,
            FloatOrder::ZerosEqual => Self::ZerosEqual,
            FloatOrder::Canonical => Self::Canonical,
            FloatOrder::Ieee754 => Self::Ieee754,
            order => vortex_bail!("list membership does not support the float order {order}"),
        })
    }

    /// Whether this normalization leaves each value unchanged.
    pub(super) fn is_identity(self) -> bool {
        self == Self::default()
    }

    /// A number below [`Self::COUNT`] that identifies this normalization. The identity is `0`.
    pub(super) fn index(self) -> usize {
        match self {
            Self::Identity => 0,
            Self::ZerosEqual => 1,
            Self::Canonical => 2,
            Self::Ieee754 => 3,
        }
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
            return match (self, side) {
                (Self::Identity | Self::ZerosEqual, _) => value,
                (Self::Canonical, _) | (Self::Ieee754, Side::Element) => T::nan(),
                // A needle `NaN` gets a bit pattern that no element has, so it is in no list.
                (Self::Ieee754, Side::Needle) => -T::nan(),
            };
        }

        if !self.is_identity() && value.is_zero() {
            return T::zero();
        }

        value
    }
}
