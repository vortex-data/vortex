//  SPDX-License-Identifier: Apache-2.0
//  SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Set-based operations and helpers

use std::hash::Hash;

use hashbrown::DefaultHashBuilder;

use crate::aliases::hash_set::HashSet;

/// Trait for performing unique counts, using the preferred hash builder.
pub trait UniqueCount {
    /// Count the number of unique elements in the iterator.
    fn unique_count(self) -> usize;
}

impl<IntoIter, Item> UniqueCount for IntoIter
where
    IntoIter: IntoIterator<Item = Item>,
    Item: Eq + Hash,
{
    fn unique_count(self) -> usize {
        unique_count(self)
    }
}

/// Count the unique elements in the iterator.
pub fn unique_count<Item, I>(iter: I) -> usize
where
    Item: Eq + Hash,
    I: IntoIterator<Item = Item>,
{
    HashSet::<_, DefaultHashBuilder>::from_iter(iter).len()
}
