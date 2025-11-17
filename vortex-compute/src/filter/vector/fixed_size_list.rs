// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_vector::fixed_size_list::{FixedSizeListVector, FixedSizeListVectorMut};

use crate::filter::Filter;

// TODO(aduffy): there really isn't a cheap way to implement these is there.

impl<M> Filter<M> for &FixedSizeListVector {
    type Output = FixedSizeListVector;

    fn filter(self, _selection: &M) -> Self::Output {
        // We need to spread the mask out to point to offsets from
        // the inner vector type
        todo!()
    }
}

impl<M> Filter<M> for &mut FixedSizeListVectorMut {
    type Output = ();

    fn filter(self, _selection: &M) -> Self::Output {
        // We need to spread the mask out to point to offsets from
        // the inner vector type
        todo!()
    }
}
