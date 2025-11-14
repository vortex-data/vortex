// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::cow::{Cow, Moo};
use vortex_buffer::{Buffer, BufferMut};

impl<T> Moo for Buffer<T> {
    type Mut = BufferMut<T>;

    fn into_mut(self) -> Self::Mut {
        Buffer::<T>::into_mut(self)
    }

    fn freeze(mutable: Self::Mut) -> Self {
        BufferMut::<T>::freeze(mutable)
    }
}

pub trait BufferOps {
    fn len(&self) -> usize;
}

impl<T> BufferOps for Cow<Buffer<T>> {
    fn len(&self) -> usize {
        match self {
            Cow::Owned(immutable) => immutable.len(),
            Cow::OwnedMut(mutable) => mutable.len(),
        }
    }
}
