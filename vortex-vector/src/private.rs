// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This private module contains the [`Sealed`] implementations for different [`Vector`] types. This
//! allows us to seal our [`VectorOps`] and [`VectorOps`] traits.
//!
//! Sealing these traits prevents external crates from implementing them while still allowing public
//! usage, which gives us the freedom to add new trait methods in the future without breaking
//! backward compatibility.

use crate::binaryview::{BinaryViewScalar, BinaryViewType, BinaryViewVector};
use crate::bool::{BoolScalar, BoolVector};
use crate::decimal::{DScalar, DVector, DecimalScalar, DecimalVector};
use crate::fixed_size_list::{FixedSizeListScalar, FixedSizeListVector};
use crate::listview::{ListViewScalar, ListViewVector};
use crate::null::{NullScalar, NullVector};
use crate::primitive::{PScalar, PVector, PrimitiveScalar, PrimitiveVector};
use crate::struct_::{StructScalar, StructVector};
use crate::{Datum, Scalar, Vector};
use vortex_dtype::{NativeDecimalType, NativePType};

/// A private trait for sealing implementations of other traits.
pub trait Sealed {}

impl Sealed for Vector {}

impl Sealed for NullVector {}
impl Sealed for BoolVector {}
impl Sealed for DecimalVector {}
impl<D: NativeDecimalType> Sealed for DVector<D> {}
impl Sealed for PrimitiveVector {}
impl<T: NativePType> Sealed for PVector<T> {}
impl<T: BinaryViewType> Sealed for BinaryViewVector<T> {}
impl Sealed for FixedSizeListVector {}
impl Sealed for ListViewVector {}
impl Sealed for StructVector {}

impl Sealed for Scalar {}
impl Sealed for NullScalar {}
impl Sealed for BoolScalar {}
impl Sealed for DecimalScalar {}
impl<D: NativeDecimalType> Sealed for DScalar<D> {}
impl Sealed for PrimitiveScalar {}
impl<T: NativePType> Sealed for PScalar<T> {}
impl<T: BinaryViewType> Sealed for BinaryViewScalar<T> {}
impl Sealed for ListViewScalar {}
impl Sealed for FixedSizeListScalar {}
impl Sealed for StructScalar {}

impl Sealed for Datum {}
