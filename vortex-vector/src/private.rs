// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This private module contains the [`Sealed`] implementations for different [`Vector`] types. This
//! allows us to seal our [`VectorOps`] and [`VectorMutOps`] traits.
//!
//! Sealing these traits prevents external crates from implementing them while still allowing public
//! usage, which gives us the freedom to add new trait methods in the future without breaking
//! backward compatibility.

use vortex_dtype::NativePType;

use crate::*;

/// A private trait for sealing implementations of other traits.
pub trait Sealed {}

impl Sealed for Vector {}
impl Sealed for VectorMut {}

impl Sealed for NullVector {}
impl Sealed for NullVectorMut {}

impl Sealed for BoolVector {}
impl Sealed for BoolVectorMut {}

impl Sealed for PrimitiveVector {}
impl Sealed for PrimitiveVectorMut {}
impl<T: NativePType> Sealed for PVector<T> {}
impl<T: NativePType> Sealed for PVectorMut<T> {}

impl Sealed for StructVector {}
impl Sealed for StructVectorMut {}
