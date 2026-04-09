// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// Slot positions for TurboQuantArray children.
///
/// Norms are not stored in the TurboQuantArray. They live in the external [`L2Denorm`]
/// ScalarFnArray wrapper returned by [`turboquant_encode`].
///
/// [`L2Denorm`]: crate::scalar_fns::l2_denorm::L2Denorm
/// [`turboquant_encode`]: crate::encodings::turboquant::turboquant_encode
#[repr(usize)]
#[derive(Clone, Copy, Debug)]
pub(crate) enum Slot {
    Codes = 0,
    Centroids = 1,
    RotationSigns = 2,
}

impl Slot {
    pub(crate) const COUNT: usize = 3;

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Codes => "codes",
            Self::Centroids => "centroids",
            Self::RotationSigns => "rotation_signs",
        }
    }

    pub(crate) fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Codes,
            1 => Self::Centroids,
            2 => Self::RotationSigns,
            _ => vortex_error::vortex_panic!("invalid slot index {idx}"),
        }
    }
}
