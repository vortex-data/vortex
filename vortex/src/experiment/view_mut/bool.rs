// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::N;
use crate::experiment::selection::Selection;
use crate::experiment::view_mut::{BitVector, VType, ViewMut};
use bitvec::order::Msb0;
use bitvec::slice::BitSlice;
use bitvec::view::BitViewSized;
use vortex_error::{VortexExpect, vortex_err};

impl<'v> ViewMut<'v> {
    pub fn new_bool(values: &'v mut BitVector, validity: Option<&'v mut BitVector>) -> Self {
        ViewMut {
            vtype: VType::Bool,
            elements: values.as_raw_mut().as_mut_ptr().cast(),
            validity,
            selection: Selection::default(),
            data: vec![],
            children: vec![],
            _marker: Default::default(),
        }
    }

    /// Access this vector as bool.
    pub fn as_bool<'a>(&'a mut self) -> BoolVector<'a, 'v> {
        assert_eq!(self.vtype, VType::Bool, "Invalid type for primitive view");
        BoolVector { view: self }
    }
}

pub struct BoolVector<'a, 'v> {
    view: &'a mut ViewMut<'v>,
}

impl<'a, 'v> BoolVector<'a, 'v> {
    // /// Return a [`BitMaskView`] over the elements of this boolean vector.
    // ///
    // /// ## Panics
    // ///
    // /// Panics if the vector is non-constant and not flat. Recommended to call flatten first.
    // pub fn as_mask(&self) -> BitMaskView {
    //     match self.as_constant() {
    //         None => {
    //             assert!(
    //                 self.view.is_flat(),
    //                 "Cannot return mask for non-flat vector"
    //             );
    //             BitMaskView::Some(self.as_ref())
    //         }
    //         Some(true) => BitMaskView::All,
    //         Some(false) => BitMaskView::None,
    //     }
    // }
    //
    // /// Return the constant value of the vector if it is a constant selection.
    // pub fn as_constant(&self) -> Option<bool> {
    //     if let Selection::Constant { element, len } = self.view.selection {
    //         self.as_ref().get(element).as_deref().copied()
    //     } else {
    //         None
    //     }
    // }
}

impl AsRef<BitVector> for BoolVector<'_, '_> {
    fn as_ref(&self) -> &BitVector {
        let ptr = self.view.elements.cast::<u64>();
        let slice = unsafe {
            // SAFETY: We assume that the elements are of type u64 and that the view is valid.
            std::slice::from_raw_parts(ptr, N / 64)
        };
        todo!();
    }
}

/// Provide mutable access to the bit elements of a boolean vector.
impl AsMut<BitVector> for BoolVector<'_, '_> {
    fn as_mut(&mut self) -> &mut BitVector {
        let ptr = self.view.elements.cast::<u64>();
        let slice = unsafe {
            // SAFETY: We assume that the elements are of type u64 and that the view is valid.
            std::slice::from_raw_parts_mut(ptr, N / 64)
        };
        todo!();
    }
}
