// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TurboQuant array definition: stores quantized coordinate codes, norms,
//! centroids (codebook), and rotation signs.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::stats::ArrayStats;
use vortex_array::vtable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

/// Encoding marker type for TurboQuant.
#[derive(Clone, Debug)]
pub struct TurboQuant;

impl TurboQuant {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.turboquant");
}

vtable!(TurboQuant, TurboQuant, TurboQuantData);

/// Protobuf metadata for TurboQuant encoding.
#[derive(Clone, prost::Message)]
pub struct TurboQuantMetadata {
    /// Vector dimension d.
    #[prost(uint32, tag = "1")]
    pub dimension: u32,
    /// MSE bits per coordinate (1-8).
    #[prost(uint32, tag = "2")]
    pub bit_width: u32,
}

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

/// TurboQuant array.
///
/// Slots:
/// - 0: `codes` -- `FixedSizeListArray<u8>` (quantized indices, list_size=padded_dim).
/// - 1: `norms` -- `PrimitiveArray<f32>` (one per vector row).
/// - 2: `centroids` -- `PrimitiveArray<f32>` (codebook, length 2^bit_width).
/// - 3: `rotation_signs` -- `BitPackedArray` (3 * padded_dim, 1-bit u8 0/1, inverse order).
#[derive(Clone, Debug)]
pub struct TurboQuantData {
    pub(crate) dtype: DType,
    pub(crate) slots: Vec<Option<ArrayRef>>,
    pub(crate) dimension: u32,
    pub(crate) bit_width: u8,
    pub(crate) stats_set: ArrayStats,
}

impl TurboQuantData {
    /// Build a TurboQuant array.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        dtype: DType,
        codes: ArrayRef,
        norms: ArrayRef,
        centroids: ArrayRef,
        rotation_signs: ArrayRef,
        dimension: u32,
        bit_width: u8,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            (1..=8).contains(&bit_width),
            "bit_width must be 1-8, got {bit_width}"
        );
        let mut slots = vec![None; Slot::COUNT];
        slots[Slot::Codes as usize] = Some(codes);
        slots[Slot::Norms as usize] = Some(norms);
        slots[Slot::Centroids as usize] = Some(centroids);
        slots[Slot::RotationSigns as usize] = Some(rotation_signs);
        Ok(Self {
            dtype,
            slots,
            dimension,
            bit_width,
            stats_set: Default::default(),
        })
    }

    /// The vector dimension d.
    pub fn dimension(&self) -> u32 {
        self.dimension
    }

    /// MSE bits per coordinate.
    pub fn bit_width(&self) -> u8 {
        self.bit_width
    }

    /// Padded dimension (next power of 2 >= dimension).
    pub fn padded_dim(&self) -> u32 {
        self.dimension.next_power_of_two()
    }

    fn slot(&self, idx: usize) -> &ArrayRef {
        self.slots[idx]
            .as_ref()
            .vortex_expect("required slot is None")
    }

    /// The quantized codes child (FixedSizeListArray).
    pub fn codes(&self) -> &ArrayRef {
        self.slot(Slot::Codes as usize)
    }

    /// The norms child (`PrimitiveArray<f32>`).
    pub fn norms(&self) -> &ArrayRef {
        self.slot(Slot::Norms as usize)
    }

    /// The centroids (codebook) child (`PrimitiveArray<f32>`).
    pub fn centroids(&self) -> &ArrayRef {
        self.slot(Slot::Centroids as usize)
    }

    /// The MSE rotation signs child (BitPackedArray, length 3 * padded_dim).
    pub fn rotation_signs(&self) -> &ArrayRef {
        self.slot(Slot::RotationSigns as usize)
    }
}
