// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// A clone-on-write enum that can hold either an owned immutable or owned mutable value.
#[derive(Debug)]
pub enum Cow<T: Moo> {
    /// An owned immutable value.
    Owned(T),
    /// An owned mutable value.
    OwnedMut(T::Mut),
}

/// A type that has both mutable and immutable forms, with conversion methods between them.
///
/// Use by the [`Cow`] enum.
pub trait Moo {
    /// The mutable form of this type.
    type Mut;

    /// Convert the immutable value into a mutable one.
    fn into_mut(self) -> Self::Mut;

    /// Convert the mutable value into an immutable one.
    fn freeze(mutable: Self::Mut) -> Self;
}
