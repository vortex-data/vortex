pub type HashMap<K, V> = hashbrown::HashMap<K, V>;
pub type Entry<'a, K, V, S> = hashbrown::hash_map::Entry<'a, K, V, S>;
pub type IntoIter<K, V> = hashbrown::hash_map::IntoIter<K, V>;
