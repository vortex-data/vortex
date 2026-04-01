// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::dtype::DType;
use crate::stats::ArrayStats;
use crate::validity::Validity;
use crate::vtable::validity_to_child;

pub(super) const CHILD_SLOT: usize = 0;
pub(super) const VALIDITY_SLOT: usize = 1;
pub(super) const NUM_SLOTS: usize = 2;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["child", "validity"];

#[derive(Clone, Debug)]
pub struct MaskedArray {
    pub(super) slots: Vec<Option<ArrayRef>>,
    pub(super) validity: Validity,
    pub(super) dtype: DType,
    pub(super) stats: ArrayStats,
}

impl MaskedArray {
    pub fn try_new(child: ArrayRef, validity: Validity) -> VortexResult<Self> {
        if matches!(validity, Validity::NonNullable) {
            vortex_bail!("MaskedArray must have nullable validity, got {validity:?}")
        }

        if !child.all_valid()? {
            vortex_bail!("MaskedArray children must not have nulls");
        }

        if let Some(validity_len) = validity.maybe_len()
            && validity_len != child.len()
        {
            vortex_bail!("Validity must be the same length as a MaskedArray's child");
        }

        // MaskedArray's nullability is determined solely by its validity, not the child's dtype.
        // The child can have nullable dtype but must not have any actual null values.
        let dtype = child.dtype().as_nullable();
        let len = child.len();
        let validity_slot = validity_to_child(&validity, len);

        Ok(Self {
            slots: vec![Some(child), validity_slot],
            validity,
            dtype,
            stats: ArrayStats::default(),
        })
    }

    pub fn child(&self) -> &ArrayRef {
        self.slots[CHILD_SLOT]
            .as_ref()
            .vortex_expect("MaskedArray child slot")
    }
}
