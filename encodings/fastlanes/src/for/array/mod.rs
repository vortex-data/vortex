// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

pub mod for_compress;
pub mod for_decompress;

/// The encoded array with the frame-of-reference (minimum value) subtracted.
pub(super) const ENCODED_SLOT: usize = 0;
pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["encoded"];

/// Frame of Reference (FoR) encoded array.
///
/// This encoding stores values as offsets from a reference value, which can significantly reduce
/// storage requirements when values are clustered around a specific point.
#[derive(Clone, Debug)]
pub struct FoRData {
    pub(super) slots: Vec<Option<ArrayRef>>,
    pub(super) reference: Scalar,
}

impl FoRData {
    pub(crate) fn try_new(encoded: ArrayRef, reference: Scalar) -> VortexResult<Self> {
        Self::validate_parts(&encoded, &reference, reference.dtype(), encoded.len())?;

        Ok(Self {
            slots: vec![Some(encoded)],
            reference,
        })
    }

    pub(crate) fn validate(&self, dtype: &DType, len: usize) -> VortexResult<()> {
        Self::validate_parts(self.encoded(), &self.reference, dtype, len)
    }

    fn validate_parts(
        encoded: &ArrayRef,
        reference: &Scalar,
        dtype: &DType,
        len: usize,
    ) -> VortexResult<()> {
        vortex_ensure!(!reference.is_null(), "Reference value cannot be null");
        vortex_ensure!(dtype.is_int(), "FoR requires an integer dtype, got {dtype}");
        vortex_ensure!(
            reference.dtype() == dtype,
            "FoR reference dtype mismatch: expected {dtype}, got {}",
            reference.dtype()
        );
        vortex_ensure!(
            encoded.dtype() == dtype,
            "FoR encoded dtype mismatch: expected {dtype}, got {}",
            encoded.dtype()
        );
        vortex_ensure!(
            encoded.len() == len,
            "FoR encoded length mismatch: expected {len}, got {}",
            encoded.len()
        );
        Ok(())
    }

    /// Returns the length of the array.
    #[inline]
    pub fn len(&self) -> usize {
        self.encoded().len()
    }

    /// Returns `true` if the array is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.encoded().is_empty()
    }

    /// Returns the dtype of the array.
    #[inline]
    pub fn dtype(&self) -> &DType {
        self.reference.dtype()
    }

    #[inline]
    pub fn ptype(&self) -> PType {
        self.dtype().as_ptype()
    }

    #[inline]
    pub fn encoded(&self) -> &ArrayRef {
        self.slots[ENCODED_SLOT]
            .as_ref()
            .vortex_expect("FoRArray encoded slot")
    }

    #[inline]
    pub fn reference_scalar(&self) -> &Scalar {
        &self.reference
    }
}
