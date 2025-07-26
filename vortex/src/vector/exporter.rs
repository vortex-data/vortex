// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use bitvec::prelude::BitVec;
use std::marker::PhantomData;

/// An exporter is the interface for exporting data from an evaluation in canonical form.
///
/// Rather than pass a mut reference to an `Evaluation`, we pass an owned `Exporter` struct such
/// that we can take ownership of the exporter in the API to prevent accidental double exports!
pub struct Exporter<'a> {
    _marker: PhantomData<&'a ()>,
}

impl<'a> Exporter<'a> {
    pub fn boolean(self) -> BooleanExporter<'a> {
        BooleanExporter {
            _marker: PhantomData,
        }
    }

    pub fn primitive<T>(self) -> PrimitiveExporter<'a, T> {
        PrimitiveExporter {
            _marker: PhantomData,
        }
    }
}

pub struct BooleanExporter<'a> {
    _marker: PhantomData<&'a ()>,
}

pub struct PrimitiveExporter<'a, T> {
    _marker: PhantomData<&'a T>,
}

impl<'a, T> PrimitiveExporter<'a, T> {
    pub fn validity_mut(&mut self) -> &mut BitVec {
        todo!()
    }
}

/// Provide access to the underlying primitive data as a slice.
impl<'a, T> AsMut<[T]> for PrimitiveExporter<'a, T> {
    fn as_mut(&mut self) -> &mut [T] {
        todo!()
    }
}

pub struct StructExporter<'a, T> {
    _marker: PhantomData<&'a T>,
}

pub struct ListExporter<'a, T> {
    _marker: PhantomData<&'a T>,
}
