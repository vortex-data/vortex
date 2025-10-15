// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Vector types for Vortex.

#![deny(missing_docs)]

use vortex_dtype::DType;

use crate::ops::{VectorMutOps, VectorOps};

mod bool;
mod bool_mut;
mod null;
mod null_mut;
mod ops;
mod primitive;
mod primitive_mut;
mod pvector;

pub use bool::*;
pub use bool_mut::*;
pub use null::*;
pub use null_mut::*;
pub use primitive::*;
pub use primitive_mut::*;
pub use pvector::*;
use vortex_error::vortex_panic;

/// An enum over all vector types.
pub enum Vector {
    /// Null
    Null(NullVector),
    /// Bool
    Bool(BoolVector),
    /// Primitive
    Primitive(PVector),
    // Decimal
    // Decimal(DecimalVector),
    // String
    // String(StringVector),
    // Binary
    // Binary(BinaryVector),
    // List
    // List(ListVector),
    // FixedList
    // FixedList(FixedListVector),
    // Struct
    // Struct(StructVector),
    // Extension
    // Extension(ExtensionVector),
}

/// An enum over all mutable vector types.
pub enum VectorMut {
    /// Null
    Null(NullVectorMut),
    /// Bool
    Bool(BoolVectorMut),
    /// Primitive
    Primitive(PVectorMut),
}

impl VectorMut {
    /// Create a new mutable vector with the given capacity and dtype.
    pub fn with_capacity(capacity: usize, dtype: &DType) -> Self {
        use vortex_dtype::NativePType;
        match dtype {
            DType::Null => NullVectorMut::new(0).into(),
            DType::Bool(n) => BoolVectorMut::with_capacity(capacity, *n).into(),
            DType::Primitive(ptype, nullability) => {
                PVectorMut::with_capacity(capacity, *ptype, *nullability).into()
            }
            _ => vortex_panic!("Unsupported dtype for VectorMut"),
        }
    }
}

macro_rules! match_each_vector {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            Vector::Null(v) => {
                let $vec = v;
                $body
            }
            Vector::Bool(v) => {
                let $vec = v;
                $body
            }
            Vector::Primitive(v) => {
                let $vec = v;
                $body
            }
        }
    }};
}

macro_rules! match_each_vector_mut {
    ($self:expr, | $vec:ident | $body:block) => {{
        match $self {
            VectorMut::Null(v) => {
                let $vec = v;
                $body
            }
            VectorMut::Bool(v) => {
                let $vec = v;
                $body
            }
            VectorMut::Primitive(v) => {
                let $vec = v;
                $body
            }
        }
    }};
}

macro_rules! match_each_vector_mut_pair {
    ($self:expr, $other:expr, | $vec:ident, $vec_other:ident | $body:block) => {{
        match ($self, $other) {
            (VectorMut::Null(a), VectorMut::Null(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (VectorMut::Bool(a), VectorMut::Bool(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (VectorMut::Primitive(a), VectorMut::Primitive(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            _ => vortex_panic!("Mismatched vector types"),
        }
    }};
}

macro_rules! match_each_vector_mut_immut_pair {
    ($self:expr, $other:expr, | $vec:ident, $vec_other:ident | $body:block) => {{
        match ($self, $other) {
            (VectorMut::Null(a), Vector::Null(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (VectorMut::Bool(a), Vector::Bool(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            (VectorMut::Primitive(a), Vector::Primitive(b)) => {
                let $vec = a;
                let $vec_other = b;
                $body
            }
            _ => vortex_panic!("Mismatched vector types"),
        }
    }};
}

impl VectorOps for Vector {
    type Mutable = VectorMut;

    fn len(&self) -> usize {
        match_each_vector!(self, |v| { v.len() })
    }

    fn dtype(&self) -> &DType {
        match_each_vector!(self, |v| { v.dtype() })
    }

    fn try_into_mut(self) -> Result<Self::Mutable, Self>
    where
        Self: Sized,
    {
        match_each_vector!(self, |v| {
            v.try_into_mut().map(VectorMut::from).map_err(Vector::from)
        })
    }
}

impl VectorMutOps for VectorMut {
    type Immutable = Vector;

    fn len(&self) -> usize {
        match_each_vector_mut!(self, |v| { v.len() })
    }

    fn dtype(&self) -> &DType {
        match_each_vector_mut!(self, |v| { v.dtype() })
    }

    fn capacity(&self) -> usize {
        match_each_vector_mut!(self, |v| { v.capacity() })
    }

    fn reserve(&mut self, additional: usize) {
        match_each_vector_mut!(self, |v| { v.reserve(additional) })
    }

    fn split_off(&mut self, at: usize) -> Self {
        match_each_vector_mut!(self, |v| { v.split_off(at).into() })
    }

    fn unsplit(&mut self, other: Self) {
        match_each_vector_mut_pair!(self, other, |a, b| {
            a.unsplit(b);
        });
    }

    fn extend_from_vector(&mut self, other: &Self::Immutable) {
        match_each_vector_mut_immut_pair!(self, other, |a, b| {
            a.extend_from_vector(b);
        });
    }

    fn freeze(self) -> Self::Immutable {
        match_each_vector_mut!(self, |v| { v.freeze().into() })
    }
}
