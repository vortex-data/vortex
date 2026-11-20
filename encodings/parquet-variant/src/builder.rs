// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Concatenates Parquet Variant storage arrays into one.

use std::any::Any;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::builders::VariantStorageBuilder;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;

use crate::ParquetVariant;
use crate::ParquetVariantArrayExt;
use crate::ParquetVariantArraySlotsExt;

/// Concatenates Parquet Variant storage arrays into one, whose children chunk the children of
/// each. Nothing is decoded: every child keeps the encoding it was appended in.
///
/// Every appended array must have the same children. Their dtypes may differ in nullability, in
/// which case the child is nullable in the result.
pub(crate) struct StorageBuilder {
    nullability: Nullability,
    len: usize,
    validity: Vec<(Validity, usize)>,
    metadata: Vec<ArrayRef>,
    value: Vec<Option<ArrayRef>>,
    typed_value: Vec<Option<ArrayRef>>,
}

impl StorageBuilder {
    /// Creates a builder of storage for Variants of `dtype`.
    pub(crate) fn new(dtype: &DType) -> Self {
        Self {
            nullability: dtype.nullability(),
            len: 0,
            validity: Vec::new(),
            metadata: Vec::new(),
            value: Vec::new(),
            typed_value: Vec::new(),
        }
    }

    /// Appends the rows of `storage`.
    pub(crate) fn append(&mut self, storage: ArrayView<'_, ParquetVariant>) -> VortexResult<()> {
        vortex_ensure!(
            self.nullability.is_nullable() || !storage.dtype().is_nullable(),
            "cannot append nullable Variant storage to a builder of non-nullable storage"
        );
        if storage.is_empty() {
            return Ok(());
        }
        self.len += storage.len();
        self.validity
            .push((storage.parquet_variant_validity(), storage.len()));
        self.metadata.push(storage.metadata().clone());
        self.value.push(storage.value().cloned());
        self.typed_value.push(storage.typed_value().cloned());
        Ok(())
    }

    /// Finishes the builder into one Parquet Variant storage array.
    pub(crate) fn finish_storage(self) -> VortexResult<ArrayRef> {
        let validity = match self.nullability {
            Nullability::NonNullable => Validity::NonNullable,
            Nullability::Nullable
                if self.validity.iter().all(|(validity, _)| {
                    matches!(validity, Validity::NonNullable | Validity::AllValid)
                }) =>
            {
                Validity::AllValid
            }
            Nullability::Nullable => Validity::Array(chunk(
                self.validity
                    .into_iter()
                    .map(|(validity, len)| validity.to_array(len))
                    .collect(),
            )?),
        };
        let metadata = chunk_child("metadata", self.metadata.into_iter().map(Some))?
            .ok_or_else(|| vortex_err!("Parquet Variant storage always has a metadata child"))?;
        let value = chunk_child("value", self.value.into_iter())?;
        let typed_value = chunk_child("typed_value", self.typed_value.into_iter())?;
        ParquetVariant::try_new(validity, metadata, value, typed_value).map(IntoArray::into_array)
    }
}

impl VariantStorageBuilder for StorageBuilder {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn len(&self) -> usize {
        self.len
    }

    fn finish(self: Box<Self>) -> VortexResult<ArrayRef> {
        self.finish_storage()
    }
}

/// Chunks the `name` child of every appended storage, or returns `None` if none has it.
fn chunk_child(
    name: &str,
    children: impl Iterator<Item = Option<ArrayRef>>,
) -> VortexResult<Option<ArrayRef>> {
    let children = children.collect::<Vec<_>>();
    if children.iter().all(Option::is_none) {
        return Ok(None);
    }
    let children = children
        .into_iter()
        .map(|child| {
            child.ok_or_else(|| {
                vortex_err!("cannot concatenate Parquet Variant storage with and without {name}")
            })
        })
        .collect::<VortexResult<Vec<_>>>()?;

    let dtype = if children.iter().any(|child| child.dtype().is_nullable()) {
        children[0].dtype().as_nullable()
    } else {
        children[0].dtype().clone()
    };
    let children = children
        .into_iter()
        .map(|child| {
            if child.dtype() == &dtype {
                Ok(child)
            } else {
                child.cast(dtype.clone())
            }
        })
        .collect::<VortexResult<Vec<_>>>()?;
    chunk(children).map(Some)
}

/// One array of `chunks`, which share a dtype.
fn chunk(mut chunks: Vec<ArrayRef>) -> VortexResult<ArrayRef> {
    if chunks.len() == 1 {
        return Ok(chunks.remove(0));
    }
    let dtype = chunks[0].dtype().clone();
    Ok(ChunkedArray::try_new(chunks, dtype)?.into_array())
}
