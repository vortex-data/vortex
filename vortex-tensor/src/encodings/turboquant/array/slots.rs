// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// Slot positions for TurboQuantArray children.
#[repr(usize)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum Slot {
    Codes = 0,
    Norms = 1,
    Centroids = 2,
    RotationSigns = 3,
}

impl Slot {
    pub(crate) const COUNT: usize = 4;

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Codes => "codes",
            Self::Norms => "norms",
            Self::Centroids => "centroids",
            Self::RotationSigns => "rotation_signs",
        }
    }

    pub(crate) fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Codes,
            1 => Self::Norms,
            2 => Self::Centroids,
            3 => Self::RotationSigns,
            _ => vortex_error::vortex_panic!("invalid slot index {idx}"),
        }
    }
}
