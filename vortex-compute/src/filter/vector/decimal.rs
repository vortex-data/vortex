// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::i256;
use vortex_vector::decimal::{DVector, DVectorMut, DecimalVector, DecimalVectorMut};
use vortex_vector::{match_each_dvector, match_each_dvector_mut};

use crate::filter::Filter;

impl<M> Filter<M> for &DecimalVector
where
    for<'a> &'a DVector<i8>: Filter<M, Output = DVector<i8>>,
    for<'a> &'a DVector<i16>: Filter<M, Output = DVector<i16>>,
    for<'a> &'a DVector<i32>: Filter<M, Output = DVector<i32>>,
    for<'a> &'a DVector<i64>: Filter<M, Output = DVector<i64>>,
    for<'a> &'a DVector<i128>: Filter<M, Output = DVector<i128>>,
    for<'a> &'a DVector<i256>: Filter<M, Output = DVector<i256>>,
{
    type Output = DecimalVector;

    fn filter(self, selection: &M) -> Self::Output {
        match_each_dvector!(self, |d| { d.filter(selection).into() })
    }
}

impl<M> Filter<M> for &mut DecimalVectorMut
where
    for<'a> &'a mut DVectorMut<i8>: Filter<M, Output = ()>,
    for<'a> &'a mut DVectorMut<i16>: Filter<M, Output = ()>,
    for<'a> &'a mut DVectorMut<i32>: Filter<M, Output = ()>,
    for<'a> &'a mut DVectorMut<i64>: Filter<M, Output = ()>,
    for<'a> &'a mut DVectorMut<i128>: Filter<M, Output = ()>,
    for<'a> &'a mut DVectorMut<i256>: Filter<M, Output = ()>,
{
    type Output = ();

    fn filter(self, selection: &M) -> Self::Output {
        match_each_dvector_mut!(self, |d| { d.filter(selection) });
    }
}
