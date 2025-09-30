// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::{BitAnd, BitOr, Not};

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::{StructArray, StructVTable};
use crate::compute::{ZipKernel, ZipKernelAdapter, zip};
use crate::validity::Validity;
use crate::vtable::ValidityHelper;
use crate::{Array, ArrayRef, register_kernel};

impl ZipKernel for StructVTable {
    fn zip(
        &self,
        if_true: &StructArray,
        if_false: &dyn Array,
        mask: &Mask,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(if_false) = if_false.as_opt::<StructVTable>() else {
            return Ok(None);
        };

        assert_eq!(if_true.names(), if_false.names()); // this is properly checked above

        let fields = if_true
            .fields()
            .iter()
            .zip(if_false.fields().iter())
            .map(|(t, f)| zip(t, f, mask))
            .collect::<VortexResult<Vec<_>>>()?;

        let validity = match (if_true.validity(), if_false.validity()) {
            (&Validity::NonNullable, &Validity::NonNullable) => Validity::NonNullable,
            (&Validity::AllValid, &Validity::AllValid) => Validity::AllValid,
            (&Validity::AllInvalid, &Validity::AllInvalid) => Validity::AllInvalid,

            (v1, v2) => {
                let v1m = v1.to_mask(if_true.len());
                let v2m = v2.to_mask(if_false.len());

                let combined = (v1m.bitand(mask)).bitor(&v2m.bitand(&mask.not()));
                Validity::from_mask(
                    combined,
                    if_true.dtype.nullability() | if_false.dtype.nullability(),
                )
            }
        };

        Ok(Some(
            StructArray::try_new(if_true.names().clone(), fields, if_true.len(), validity)?
                .to_array(),
        ))
    }
}

register_kernel!(ZipKernelAdapter(StructVTable).lift());
