use itertools::Itertools;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};

use crate::arrays::{StructArray, StructVTable};
use crate::compute::{CastKernel, CastKernelAdapter, cast};
use crate::vtable::ValidityHelper;
use crate::{ArrayRef, IntoArray, register_kernel};

impl CastKernel for StructVTable {
    fn cast(&self, array: &StructArray, dtype: &DType) -> VortexResult<ArrayRef> {
        let Some(target_sdtype) = dtype.as_struct() else {
            vortex_bail!("cannot cast {} to {}", array.dtype(), dtype);
        };

        let source_sdtype = array
            .dtype()
            .as_struct()
            .vortex_expect("struct array must have struct dtype");

        if target_sdtype.names() != source_sdtype.names() {
            vortex_bail!("cannot cast {} to {}", array.dtype(), dtype);
        }

        let validity = array
            .validity()
            .clone()
            .cast_nullability(dtype.nullability())?;

        StructArray::try_new(
            target_sdtype.names().clone(),
            array
                .fields()
                .iter()
                .zip_eq(target_sdtype.fields())
                .map(|(field, dtype)| cast(field, &dtype))
                .try_collect()?,
            array.len(),
            validity,
        )
        .map(|a| a.into_array())
    }
}

register_kernel!(CastKernelAdapter(StructVTable).lift());
