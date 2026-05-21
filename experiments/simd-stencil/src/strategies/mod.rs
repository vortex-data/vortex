// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The decode composition strategies.

pub mod aot;
pub mod fused;
pub mod materialized;

#[inline(always)]
pub(crate) fn tile_u32(s: &[u32]) -> &[u32; crate::TILE] {
    s.try_into().expect("tile-sized slice")
}

#[inline(always)]
pub(crate) fn tile_u32_mut(s: &mut [u32]) -> &mut [u32; crate::TILE] {
    s.try_into().expect("tile-sized slice")
}

#[inline(always)]
pub(crate) fn tile_u64(s: &[u64]) -> &[u64; crate::TILE] {
    s.try_into().expect("tile-sized slice")
}

#[inline(always)]
pub(crate) fn tile_u64_mut(s: &mut [u64]) -> &mut [u64; crate::TILE] {
    s.try_into().expect("tile-sized slice")
}

#[inline(always)]
pub(crate) fn tile_f64_mut(s: &mut [f64]) -> &mut [f64; crate::TILE] {
    s.try_into().expect("tile-sized slice")
}
