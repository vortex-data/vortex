// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::NativePType;
use vortex_mask::Mask;
use vortex_vector::VectorMutOps;
use vortex_vector::primitive::PVectorMut;

use crate::filter::Filter;

impl<T: NativePType> Filter for &mut PVectorMut<T> {
    type Output = ();

    fn filter(self, selection_mask: &Mask) {
        assert_eq!(
            selection_mask.len(),
            self.len(),
            "Selection mask length must equal the vector length"
        );

        unimplemented!()

        // SAFETY: We filter the two components of the vector at the same time, so the length
        // invariants remain true.
        // unsafe {
        //     self.elements_mut().filter(selection_mask);
        //     self.validity_mut().filter(selection_mask);
        // }
    }
}
