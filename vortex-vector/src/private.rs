// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This private module contains the [`Sealed`] implementations for different [`Vector`] types. This
//! allows us to seal our [`VectorOps`] and [`VectorMutOps`] traits.
//!
//! Sealing these traits prevents external crates from implementing them while still allowing public
//! usage, which gives us the freedom to add new trait methods in the future without breaking
//! backward compatibility.

use vortex_dtype::{NativeDecimalType, NativePType};

use crate::binaryview::{BinaryViewType, BinaryViewVector, BinaryViewVectorMut};
use crate::bool::{BoolVector, BoolVectorMut};
use crate::decimal::{DScalar, DVector, DVectorMut, DecimalVector, DecimalVectorMut};
use crate::fixed_size_list::{FixedSizeListVector, FixedSizeListVectorMut};
use crate::listview::{ListViewVector, ListViewVectorMut};
use crate::null::{NullVector, NullVectorMut};
use crate::primitive::{PScalar, PVector, PVectorMut, PrimitiveVector, PrimitiveVectorMut};
use crate::struct_::{StructScalar, StructVector, StructVectorMut};
use crate::*;

/// A private trait for sealing implementations of other traits.
pub(crate) trait Sealed {}

impl Sealed for Vector {}
impl Sealed for VectorMut {}

impl Sealed for NullVector {}
impl Sealed for NullVectorMut {}

impl Sealed for BoolVector {}
impl Sealed for BoolVectorMut {}

impl Sealed for DecimalVector {}
impl Sealed for DecimalVectorMut {}
impl<D: NativeDecimalType> Sealed for DVector<D> {}
impl<D: NativeDecimalType> Sealed for DVectorMut<D> {}

impl Sealed for PrimitiveVector {}
impl Sealed for PrimitiveVectorMut {}
impl<T: NativePType> Sealed for PVector<T> {}
impl<T: NativePType> Sealed for PVectorMut<T> {}

impl<T: BinaryViewType> Sealed for BinaryViewVector<T> {}
impl<T: BinaryViewType> Sealed for BinaryViewVectorMut<T> {}

impl Sealed for FixedSizeListVector {}
impl Sealed for FixedSizeListVectorMut {}

impl Sealed for ListViewVector {}
impl Sealed for ListViewVectorMut {}

impl Sealed for StructVector {}
impl Sealed for StructVectorMut {}

impl Sealed for Scalar {}
impl Sealed for DScalar {}
impl Sealed for PScalar {}
impl Sealed for StructScalar {}
