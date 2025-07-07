// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub type HashSet<V, S = super::DefaultHashBuilder> = hashbrown::HashSet<V, S>;
pub type Entry<'a, V, S> = hashbrown::hash_set::Entry<'a, V, S>;
pub type IntoIter<V> = hashbrown::hash_set::IntoIter<V>;
