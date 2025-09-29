// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_scalar::Scalar;

use crate::builders::ArrayBuilder;
use crate::compute::mask;
use crate::stats::{Precision, Stat, StatsSetRef};
use crate::validity::Validity;
use crate::vtable::{
    ArrayVTable, CanonicalVTable, NotSupported, OperationsVTable, VTable, ValidityHelper,
    ValidityVTableFromValidityHelper, VisitorVTable,
};
use crate::{
    ArrayBufferVisitor, ArrayChildVisitor, ArrayRef, Canonical, EncodingId, EncodingRef, IntoArray,
    vtable,
};

vtable!(Masked);

#[derive(Clone, Debug)]
pub struct MaskedEncoding;

impl VTable for MaskedVTable {
    type Array = MaskedArray;
    type Encoding = MaskedEncoding;

    type ArrayVTable = Self;
    type CanonicalVTable = Self;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromValidityHelper;
    type VisitorVTable = Self;
    type ComputeVTable = NotSupported;
    type EncodeVTable = NotSupported;
    type PipelineVTable = NotSupported;
    type SerdeVTable = NotSupported;

    fn id(_encoding: &Self::Encoding) -> EncodingId {
        EncodingId::new_ref("vortex.masked")
    }

    fn encoding(_array: &Self::Array) -> EncodingRef {
        EncodingRef::new_ref(MaskedEncoding.as_ref())
    }
}

#[derive(Clone, Debug)]
pub struct MaskedArray {
    child: ArrayRef,
    validity: Validity,
    dtype: DType,
}

impl MaskedArray {
    fn masked_child(&self) -> VortexResult<ArrayRef> {
        mask(&self.child, &self.validity.to_mask(self.len()))
    }

    fn nullability(&self) -> Nullability {
        match self.validity {
            Validity::NonNullable => Nullability::NonNullable,
            _ => Nullability::Nullable,
        }
    }
}

impl ArrayVTable<MaskedVTable> for MaskedVTable {
    fn len(array: &MaskedArray) -> usize {
        array.child.len()
    }

    fn dtype(array: &MaskedArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &MaskedArray) -> StatsSetRef<'_> {
        if array.all_valid() {
            array.child.statistics()
        } else {
            let stats = array.child.statistics();
            for stat in Stat::all() {
                stats.clear(stat);
            }

            let null_count = array.validity.to_mask(array.len()).false_count();
            stats.set(Stat::NullCount, Precision::exact(null_count));

            stats
        }
    }
}

impl ValidityHelper for MaskedArray {
    fn validity(&self) -> &Validity {
        &self.validity
    }
}

impl MaskedArray {
    pub fn try_new(child: ArrayRef, validity: Validity) -> VortexResult<Self> {
        if child.dtype().is_nullable() {
            vortex_bail!("MaskedArray only supports non-nullable children");
        }

        if let Validity::Array(arr) = &validity
            && arr.len() != child.len()
        {
            vortex_bail!("Validity must be the same length as a MaskedArray's child");
        }

        let nullability = match validity {
            Validity::NonNullable => Nullability::NonNullable,
            _ => Nullability::Nullable,
        };

        let dtype = child.dtype().with_nullability(nullability);

        Ok(Self {
            child,
            validity,
            dtype,
        })
    }
}

impl CanonicalVTable<MaskedVTable> for MaskedVTable {
    fn canonicalize(array: &MaskedArray) -> Canonical {
        array
            .masked_child()
            .vortex_expect("Trust me")
            .to_canonical()
    }

    fn append_to_builder(array: &MaskedArray, builder: &mut dyn ArrayBuilder) {
        let child = array
            .masked_child()
            .vortex_expect("Trust me")
            .to_canonical();
        builder.extend_from_array(child.as_ref())
    }
}

impl VisitorVTable<MaskedVTable> for MaskedVTable {
    fn visit_buffers(_array: &MaskedArray, _visitor: &mut dyn ArrayBufferVisitor) {}

    fn visit_children(array: &MaskedArray, visitor: &mut dyn ArrayChildVisitor) {
        visitor.visit_child("child", array.child.as_ref());
    }
}

impl OperationsVTable<MaskedVTable> for MaskedVTable {
    fn slice(array: &MaskedArray, range: std::ops::Range<usize>) -> ArrayRef {
        let child = array.child.slice(range.clone());
        let validity = array.validity.slice(range);

        MaskedArray {
            child,
            validity,
            dtype: array.dtype.clone(),
        }
        .into_array()
    }

    fn scalar_at(array: &MaskedArray, index: usize) -> Scalar {
        let child = array.child.scalar_at(index);

        if matches!(array.nullability(), Nullability::Nullable) {
            child.into_nullable()
        } else {
            child
        }
    }
}
