// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::cow::{Cow, Moo};
use vortex_buffer::{BitBuffer, BitBufferMut};

impl Moo for BitBuffer {
    type Mut = BitBufferMut;

    fn into_mut(self) -> Self::Mut {
        BitBuffer::into_mut(self)
    }

    fn freeze(mutable: Self::Mut) -> Self {
        BitBufferMut::freeze(mutable)
    }
}

pub trait BitBufferOps {
    fn len(&self) -> usize;
}

impl BitBufferOps for Cow<BitBuffer> {
    fn len(&self) -> usize {
        match self {
            Cow::Owned(immutable) => immutable.len(),
            Cow::OwnedMut(mutable) => mutable.len(),
        }
    }
}
