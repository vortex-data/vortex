// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::aliases::hash_map::RandomState;

/// DashMap type alias using the default hash builder.
pub type DashMap<K, V> = dashmap::DashMap<K, V, RandomState>;
/// Entry type for HashMap.
pub type Entry<'a, K, V> = dashmap::Entry<'a, K, V>;
