// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fast non-cryptographic hashing for the encoder's hot integer-keyed maps,
//! mirroring upstream `include/onpair/encoding/fast_hash.h`.

use std::hash::BuildHasherDefault;
use std::hash::Hasher;

/// Multiply-xorshift hasher for small integer keys. One multiply and two
/// shifts give full avalanche into the high bits SwissTable-style maps use
/// for their control tags.
#[derive(Default)]
pub(crate) struct FastHasher {
    hash: u64,
}

const MULTIPLIER: u64 = 0xd6e8_feb8_6659_fd93;

#[inline]
fn mix(x: u64) -> u64 {
    let mut x = x ^ (x >> 32);
    x = x.wrapping_mul(MULTIPLIER);
    x ^ (x >> 32)
}

impl Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.hash = mix(self.hash.rotate_left(8) ^ u64::from(byte));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.write_u64(u64::from(i));
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.write_u64(u64::from(i));
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.hash = mix(self.hash ^ i);
    }
}

/// [`std::hash::BuildHasher`] for [`FastHasher`].
pub(crate) type FastBuildHasher = BuildHasherDefault<FastHasher>;

/// Hash map over small integer keys using [`FastHasher`].
pub(crate) type FastMap<K, V> = hashbrown::HashMap<K, V, FastBuildHasher>;
