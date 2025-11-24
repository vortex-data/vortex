// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::half::f16;
use vortex_vector::primitive::{PVector, PVectorMut, PrimitiveVector, PrimitiveVectorMut};
use vortex_vector::{VectorMutOps, VectorOps, match_each_pvector, match_each_pvector_mut};

use crate::filter::Filter;

impl<M> Filter<M> for &PrimitiveVector
where
    for<'a> &'a PVector<i8>: Filter<M, Output = PVector<i8>>,
    for<'a> &'a PVector<i16>: Filter<M, Output = PVector<i16>>,
    for<'a> &'a PVector<i32>: Filter<M, Output = PVector<i32>>,
    for<'a> &'a PVector<i64>: Filter<M, Output = PVector<i64>>,
    for<'a> &'a PVector<u8>: Filter<M, Output = PVector<u8>>,
    for<'a> &'a PVector<u16>: Filter<M, Output = PVector<u16>>,
    for<'a> &'a PVector<u32>: Filter<M, Output = PVector<u32>>,
    for<'a> &'a PVector<u64>: Filter<M, Output = PVector<u64>>,
    for<'a> &'a PVector<f16>: Filter<M, Output = PVector<f16>>,
    for<'a> &'a PVector<f32>: Filter<M, Output = PVector<f32>>,
    for<'a> &'a PVector<f64>: Filter<M, Output = PVector<f64>>,
{
    type Output = PrimitiveVector;

    fn filter(self, selection: &M) -> Self::Output {
        match_each_pvector!(self, |v| { v.filter(selection).into() })
    }
}

impl<M> Filter<M> for &mut PrimitiveVectorMut
where
    for<'a> &'a mut PVectorMut<i8>: Filter<M, Output = ()>,
    for<'a> &'a mut PVectorMut<i16>: Filter<M, Output = ()>,
    for<'a> &'a mut PVectorMut<i32>: Filter<M, Output = ()>,
    for<'a> &'a mut PVectorMut<i64>: Filter<M, Output = ()>,
    for<'a> &'a mut PVectorMut<u8>: Filter<M, Output = ()>,
    for<'a> &'a mut PVectorMut<u16>: Filter<M, Output = ()>,
    for<'a> &'a mut PVectorMut<u32>: Filter<M, Output = ()>,
    for<'a> &'a mut PVectorMut<u64>: Filter<M, Output = ()>,
    for<'a> &'a mut PVectorMut<f16>: Filter<M, Output = ()>,
    for<'a> &'a mut PVectorMut<f32>: Filter<M, Output = ()>,
    for<'a> &'a mut PVectorMut<f64>: Filter<M, Output = ()>,
{
    type Output = ();

    fn filter(self, selection: &M) -> Self::Output {
        match_each_pvector_mut!(self, |v| { v.filter(selection) })
    }
}

impl<M> Filter<M> for PrimitiveVector
where
    for<'a> &'a PrimitiveVector: Filter<M, Output = PrimitiveVector>,
    for<'a> &'a mut PrimitiveVectorMut: Filter<M, Output = ()>,
{
    type Output = Self;

    fn filter(self, selection: &M) -> Self {
        match self.try_into_mut() {
            // If we have exclusive access, we can perform the filter in place.
            Ok(mut vector_mut) => {
                (&mut vector_mut).filter(selection);
                vector_mut.freeze()
            }
            // Otherwise, allocate a new buffer and fill it in (delegate to the `&PrimitiveVector`
            // impl).
            Err(vector) => (&vector).filter(selection),
        }
    }
}
