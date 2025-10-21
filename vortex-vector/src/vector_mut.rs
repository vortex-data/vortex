// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::vortex_panic;

use crate::{
    BoolVectorMut, NullVectorMut, PrimitiveVectorMut, Vector, VectorMutOps, match_each_vector_mut,
    match_each_vector_mut_immut_pair, match_each_vector_mut_pair,
};

/// An enum over all kinds of mutable vectors, which represent fully decompressed (canonical) array
/// data.
///
/// Most of the behavior of `VectorMut` is described by the [`VectorMutOps`] trait. Note that
/// vectors are **always** considered as nullable, and it is the responsibility of the user to not
/// add any nullable data to a vector they want to keep as non-nullable.
///
/// The immutable equivalent of this type is [`Vector`], which implements the
/// [`VectorOps`](crate::VectorOps) trait.
#[derive(Debug, Clone)]
pub enum VectorMut {
    /// Null
    Null(NullVectorMut),
    /// Bool
    Bool(BoolVectorMut),
    /// Primitive
    Primitive(PrimitiveVectorMut),
}

impl VectorMut {
    /// Create a new mutable vector with the given capacity and dtype.
    pub fn with_capacity(capacity: usize, dtype: &DType) -> Self {
        match dtype {
            DType::Null => NullVectorMut::new(0).into(), // `NullVector` has `usize::MAX` capacity.
            DType::Bool(_) => BoolVectorMut::with_capacity(capacity).into(),
            DType::Primitive(ptype, _) => {
                PrimitiveVectorMut::with_capacity(*ptype, capacity).into()
            }
            _ => vortex_panic!("Unsupported dtype for VectorMut"),
        }
    }
}

impl VectorMutOps for VectorMut {
    type Immutable = Vector;

    fn len(&self) -> usize {
        match_each_vector_mut!(self, |v| { v.len() })
    }

    fn capacity(&self) -> usize {
        match_each_vector_mut!(self, |v| { v.capacity() })
    }

    fn reserve(&mut self, additional: usize) {
        match_each_vector_mut!(self, |v| { v.reserve(additional) })
    }

    fn extend_from_vector(&mut self, other: &Self::Immutable) {
        match_each_vector_mut_immut_pair!(self, other, |a, b| {
            a.extend_from_vector(b);
        });
    }

    fn append_nulls(&mut self, n: usize) {
        match_each_vector_mut!(self, |v| { v.append_nulls(n) })
    }

    fn freeze(self) -> Self::Immutable {
        match_each_vector_mut!(self, |v| { v.freeze().into() })
    }

    fn split_off(&mut self, at: usize) -> Self {
        match_each_vector_mut!(self, |v| { v.split_off(at).into() })
    }

    fn unsplit(&mut self, other: Self) {
        match_each_vector_mut_pair!(self, other, |a, b| {
            a.unsplit(b);
        });
    }
}
