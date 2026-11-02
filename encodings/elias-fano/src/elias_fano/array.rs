// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use vortex_array::ArrayRef;
use vortex_array::array_slots;

use super::vtable::EliasFano;

#[array_slots(EliasFano)]
pub struct EliasFanoSlots {
    /// The number of values in each bucket of high bits.
    #[slot(0)]
    pub upper: ArrayRef,
    /// The low `lower_bits` bits of each value.
    #[slot(1)]
    pub lower: ArrayRef,
}

/// The parts of an [`EliasFano`] array besides its children.
#[derive(Clone, Debug)]
pub struct EliasFanoData {
    pub(super) lower_bits: u8,
}

impl Display for EliasFanoData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "lower_bits: {}", self.lower_bits)
    }
}
