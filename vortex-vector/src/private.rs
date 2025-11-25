// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This private module contains the [`Sealed`] implementations for different [`Vector`] types. This
//! allows us to seal our [`VectorOps`] and [`VectorMutOps`] traits.
//!
//! Sealing these traits prevents external crates from implementing them while still allowing public
//! usage, which gives us the freedom to add new trait methods in the future without breaking
//! backward compatibility.

use vortex_dtype::NativeDecimalType;
use vortex_dtype::NativePType;

use crate::Datum;
use crate::Scalar;
use crate::Vector;
use crate::VectorMut;
use crate::binaryview::BinaryViewScalar;
use crate::binaryview::BinaryViewType;
use crate::binaryview::BinaryViewVector;
use crate::binaryview::BinaryViewVectorMut;
use crate::bool::BoolScalar;
use crate::bool::BoolVector;
use crate::bool::BoolVectorMut;
use crate::decimal::DScalar;
use crate::decimal::DVector;
use crate::decimal::DVectorMut;
use crate::decimal::DecimalScalar;
use crate::decimal::DecimalVector;
use crate::decimal::DecimalVectorMut;
use crate::fixed_size_list::FixedSizeListScalar;
use crate::fixed_size_list::FixedSizeListVector;
use crate::fixed_size_list::FixedSizeListVectorMut;
use crate::listview::ListViewScalar;
use crate::listview::ListViewVector;
use crate::listview::ListViewVectorMut;
use crate::null::NullScalar;
use crate::null::NullVector;
use crate::null::NullVectorMut;
use crate::primitive::PScalar;
use crate::primitive::PVector;
use crate::primitive::PVectorMut;
use crate::primitive::PrimitiveScalar;
use crate::primitive::PrimitiveVector;
use crate::primitive::PrimitiveVectorMut;
use crate::struct_::StructScalar;
use crate::struct_::StructVector;
use crate::struct_::StructVectorMut;

/// A private trait for sealing implementations of other traits.
pub trait Sealed {}

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
