// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fast non-cryptographic hashing for the longest-prefix-match maps.

/// Hasher used by the longest-prefix-match maps.
pub(crate) type MapHasher = hashbrown::DefaultHashBuilder;

/// Hash map keyed through [`MapHasher`].
pub(crate) type Map<K, V> = hashbrown::HashMap<K, V, MapHasher>;

/// An empty [`Map`].
#[inline]
pub(crate) fn map<K, V>() -> Map<K, V> {
    Map::with_hasher(MapHasher::default())
}

/// A [`Map`] preallocated for `cap` entries.
#[inline]
pub(crate) fn map_with_capacity<K, V>(cap: usize) -> Map<K, V> {
    Map::with_capacity_and_hasher(cap, MapHasher::default())
}
