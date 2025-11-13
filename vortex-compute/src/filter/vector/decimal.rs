use vortex_mask::Mask;
use vortex_vector::decimal::{DecimalVector, DecimalVectorMut};
use vortex_vector::{match_each_dvector, match_each_dvector_mut};

use crate::filter::{Filter, MaskIndices};

impl Filter<Mask> for &DecimalVector {
    type Output = DecimalVector;

    fn filter(self, selection: &Mask) -> Self::Output {
        match_each_dvector!(self, |d| { d.filter(selection).into() })
    }
}

impl Filter<MaskIndices<'_>> for &DecimalVector {
    type Output = DecimalVector;

    fn filter(self, selection: &MaskIndices) -> Self::Output {
        match_each_dvector!(self, |d| { d.filter(selection).into() })
    }
}

impl Filter<Mask> for &mut DecimalVectorMut {
    type Output = ();

    fn filter(self, selection: &Mask) -> Self::Output {
        match_each_dvector_mut!(self, |d| { d.filter(selection) });
    }
}

impl Filter<MaskIndices<'_>> for &mut DecimalVectorMut {
    type Output = ();

    fn filter(self, selection: &MaskIndices) -> Self::Output {
        match_each_dvector_mut!(self, |d| { d.filter(selection) });
    }
}
