// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// HashSet type alias using the default hash builder.
pub type HashSet<V, S = super::DefaultHashBuilder> = hashbrown::HashSet<V, S>;
/// Entry type for HashSet.
pub type Entry<'a, V, S> = hashbrown::hash_set::Entry<'a, V, S>;
/// IntoIter type for HashSet.
pub type IntoIter<V> = hashbrown::hash_set::IntoIter<V>;
