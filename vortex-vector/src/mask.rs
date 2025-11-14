// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::cow::{Cow, Moo};
use vortex_mask::{Mask, MaskMut};

impl Moo for Mask {
    type Mut = MaskMut;

    fn into_mut(self) -> Self::Mut {
        Mask::into_mut(self)
    }

    fn freeze(mutable: Self::Mut) -> Self {
        MaskMut::freeze(mutable)
    }
}

pub trait MaskOps {
    fn len(&self) -> usize;
}

impl MaskOps for Cow<Mask> {
    fn len(&self) -> usize {
        match self {
            Cow::Owned(immutable) => immutable.len(),
            Cow::OwnedMut(mutable) => mutable.len(),
        }
    }
}
