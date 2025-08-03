// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::experiment::mask::{BitMask, BitMaskView};
use crate::experiment::vector::{BitVector, N, Selection, VType, Vector};
use bitvec::access::BitSafeU64;
use bitvec::array::BitArray;
use bitvec::domain::Domain;
use bitvec::order::Msb0;
use bitvec::prelude::BitRef;
use bitvec::ptr::{Const, Mut};
use bitvec::slice::BitSlice;
use vortex_dtype::NativePType;
use vortex_error::VortexError::TryFromInt;
use vortex_error::{VortexExpect, vortex_err, vortex_panic};

impl<'v> Vector<'v> {
    pub fn new_bool(values: &'v mut BitVector, validity: Option<&'v mut BitVector>) -> Self {
        Vector {
            vtype: VType::Bool,
            elements: values.as_raw_mut_slice().as_mut_ptr().cast(),
            validity,
            selection: Selection::Empty,
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
    view: &'a mut Vector<'v>,
}

impl<'a, 'v> BoolVector<'a, 'v> {
    /// Return a [`BitMaskView`] over the elements of this boolean vector.
    ///
    /// ## Panics
    ///
    /// Panics if the vector is non-constant and not flat. Recommended to call flatten first.
    pub fn as_mask(&self) -> BitMaskView {
        match self.as_constant() {
            None => {
                assert!(
                    self.view.is_flat(),
                    "Cannot return mask for non-flat vector"
                );
                BitMaskView::Some(self.as_ref())
            }
            Some(true) => BitMaskView::All,
            Some(false) => BitMaskView::None,
        }
    }

    /// Return the constant value of the vector if it is a constant selection.
    pub fn as_constant(&self) -> Option<bool> {
        if let Selection::Constant { element, len } = self.view.selection {
            self.as_ref().get(element).as_deref().copied()
        } else {
            None
        }
    }
}

impl AsRef<BitVector> for BoolVector<'_, '_> {
    fn as_ref(&self) -> &BitVector {
        let ptr = self.view.elements.cast::<u64>();
        let slice = unsafe {
            // SAFETY: We assume that the elements are of type u64 and that the view is valid.
            std::slice::from_raw_parts(ptr, N / 64)
        };
        BitSlice::<u64, Msb0>::from_slice(slice)
            .try_into()
            .map_err(|_| vortex_err!("infallible"))
            .vortex_expect("Known to be the correct size")
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
        BitSlice::from_slice_mut(slice)
            .try_into()
            .map_err(|_| vortex_err!("infallible"))
            .vortex_expect("Known to be the correct size")
    }
}
