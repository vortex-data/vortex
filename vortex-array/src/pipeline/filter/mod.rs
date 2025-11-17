// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod buffer;

use vortex_compute::filter::Filter;
use vortex_dtype::NativePType;
use vortex_mask::MaskMut;
use vortex_vector::primitive::{PVectorMut, PrimitiveVectorMut};
use vortex_vector::{match_each_pvector_mut, VectorMut};

use crate::pipeline::BitView;

impl Filter<BitView<'_>> for &mut VectorMut {
    type Output = ();

    fn filter(self, selection: &BitView<'_>) -> Self::Output {
        // TODO(ngates): replace with macro when all vectors impl filter.
        match self {
            VectorMut::Null(_) => {}
            VectorMut::Bool(_) => {}
            VectorMut::Decimal(_) => {}
            VectorMut::Primitive(p) => {
                p.filter(selection);
            }
            VectorMut::String(_) => {}
            VectorMut::Binary(_) => {}
            VectorMut::List(_) => {}
            VectorMut::FixedSizeList(_) => {}
            VectorMut::Struct(_) => {}
        }
    }
}

impl Filter<BitView<'_>> for &mut PrimitiveVectorMut {
    type Output = ();

    fn filter(self, selection: &BitView<'_>) -> Self::Output {
        match_each_pvector_mut!(self, |v| { v.filter(selection) })
    }
}

impl<T: NativePType> Filter<BitView<'_>> for &mut PVectorMut<T> {
    type Output = ();

    fn filter(self, selection: &BitView<'_>) -> Self::Output {
        unsafe { self.elements_mut() }.as_mut().filter(selection);
        // FIXME(ngates): filter the validity...
        *unsafe { self.validity_mut() } = MaskMut::new_true(selection.true_count());
        unsafe { self.set_len(selection.true_count()) };
    }
}
