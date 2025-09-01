// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// The default hash builder for HashMap.
pub type DefaultHashBuilder = hashbrown::DefaultHashBuilder;
/// Random state for HashMap (alias for DefaultHashBuilder).
pub type RandomState = hashbrown::DefaultHashBuilder;
/// HashMap type alias using the default hash builder.
pub type HashMap<K, V, S = DefaultHashBuilder> = hashbrown::HashMap<K, V, S>;
/// Entry type for HashMap.
pub type Entry<'a, K, V, S> = hashbrown::hash_map::Entry<'a, K, V, S>;
/// IntoIter type for HashMap.
pub type IntoIter<K, V> = hashbrown::hash_map::IntoIter<K, V>;
/// HashTable type alias.
pub type HashTable<T> = hashbrown::HashTable<T>;
/// Entry type for HashTable.
pub type HashTableEntry<'a, T> = hashbrown::hash_table::Entry<'a, T>;
