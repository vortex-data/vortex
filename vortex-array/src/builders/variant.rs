// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ConstantArray;
use crate::arrays::VariantArray;
use crate::arrays::variant::VariantArraySlotsExt;
use crate::builders::ArrayBuilder;
use crate::builtins::ArrayBuiltins;
use crate::canonical::Canonical;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar::Scalar;

/// Builds the core storage of a Variant in the physical encoding of one storage array.
///
/// A [`VariantBuilder`] cannot merge the storage it receives, because the canonical Variant
/// leaves the storage encoding open. A storage encoding that can merge its arrays installs a
/// builder of its own through [`VariantBuilder::append_storage`] instead, so that consecutive
/// arrays of that encoding finish as one array of it rather than one chunk each.
pub trait VariantStorageBuilder: Send + 'static {
    /// Returns the builder as [`Any`], so that its encoding can recognise it.
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// The number of rows appended so far.
    fn len(&self) -> usize;

    /// Whether no row has been appended.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Finishes the storage, an array of the builder's Variant dtype with one row per appended row.
    fn finish(self: Box<Self>) -> VortexResult<ArrayRef>;
}

/// A part of the core storage, in row order.
enum CorePart {
    /// Storage kept in the encoding it was appended in.
    Chunk(ArrayRef),
    /// Storage merged by the builder of its encoding.
    Storage(Box<dyn VariantStorageBuilder>),
    /// A run of one scalar.
    Run(Scalar, usize),
}

impl CorePart {
    fn len(&self) -> usize {
        match self {
            CorePart::Chunk(chunk) => chunk.len(),
            CorePart::Storage(storage) => storage.len(),
            CorePart::Run(_, len) => *len,
        }
    }

    fn finish(self) -> VortexResult<ArrayRef> {
        match self {
            CorePart::Chunk(chunk) => Ok(chunk),
            CorePart::Storage(storage) => storage.finish(),
            CorePart::Run(scalar, len) => Ok(ConstantArray::new(scalar, len).into_array()),
        }
    }
}

/// A part of the shredded child, in row order.
enum ShreddedPart {
    /// Shredded values of the rows of a Variant that had a shredded child.
    Array(ArrayRef),
    /// Rows of Variants without a shredded child, whose values are all in the core storage.
    Unshredded(usize),
}

/// The builder for building a [`VariantArray`].
///
/// The core storage of the result keeps the encodings it was appended in. Consecutive storage
/// arrays of an encoding that installs a [`VariantStorageBuilder`] merge into one array of that
/// encoding; any other storage, and runs of scalars, become chunks of their own.
///
/// The shredded children of the appended Variants are concatenated. Rows from Variants without a
/// shredded child are null in it, which marks their values as held by the core storage alone; the
/// shredded child is then nullable even if the appended ones are not. Every shredded child must
/// have the same dtype, up to nullability.
pub struct VariantBuilder {
    dtype: DType,
    /// The core storage, in row order. Only the last part grows.
    core: Vec<CorePart>,
    /// The number of rows in every part of `core` but the last.
    finished_len: usize,
    shredded: Vec<ShreddedPart>,
    /// The number of rows `shredded` covers.
    shredded_len: usize,
    /// Whether the builder is appending the core storage of a canonical Variant.
    in_core: bool,
}

impl VariantBuilder {
    /// Creates a builder for Variants of `nullability`.
    pub fn new(nullability: Nullability) -> Self {
        Self {
            dtype: DType::Variant(nullability),
            core: Vec::new(),
            finished_len: 0,
            shredded: Vec::new(),
            shredded_len: 0,
            in_core: false,
        }
    }

    /// Appends the rows of a canonical Variant.
    ///
    /// The core storage is appended through its encoding, which may merge it into the storage
    /// appended before it.
    pub fn append_variant_array(
        &mut self,
        array: &VariantArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        self.check_dtype(array.dtype())?;
        if array.is_empty() {
            return Ok(());
        }
        if self.in_core {
            // A storage encoding without a storage builder of its own executes to a canonical
            // Variant over itself, and its rows are kept as they are. So are those of a core
            // storage that holds shredded values of its own.
            let chunk = if array.shredded().is_some() {
                array.clone().into_array()
            } else {
                array.core_storage().clone()
            };
            self.push(CorePart::Chunk(chunk));
            return Ok(());
        }

        if let Some(shredded) = array.shredded() {
            if let Some(ShreddedPart::Array(first)) = self
                .shredded
                .iter()
                .find(|part| matches!(part, ShreddedPart::Array(_)))
            {
                vortex_ensure!(
                    first.dtype().eq_ignore_nullability(shredded.dtype()),
                    "cannot build a Variant from Variants shredded as {} and as {}",
                    first.dtype(),
                    shredded.dtype()
                );
            }
            self.pad_shredded(self.len());
            self.shredded_len += shredded.len();
            self.shredded.push(ShreddedPart::Array(shredded.clone()));
        }
        self.in_core = true;
        let appended = array.core_storage().append_to_builder(self, ctx);
        self.in_core = false;
        appended
    }

    /// Appends rows of core storage through the [`VariantStorageBuilder`] of type `B`.
    ///
    /// If the storage appended last went through a builder of type `B`, `append` extends it.
    /// Otherwise `new` creates one for the builder's dtype.
    pub fn append_storage<B: VariantStorageBuilder>(
        &mut self,
        dtype: &DType,
        new: impl FnOnce(&DType) -> B,
        append: impl FnOnce(&mut B) -> VortexResult<()>,
    ) -> VortexResult<()> {
        self.check_dtype(dtype)?;
        let reuse = match self.core.last_mut() {
            Some(CorePart::Storage(storage)) => storage.as_any_mut().is::<B>(),
            _ => false,
        };
        if !reuse {
            self.push(CorePart::Storage(Box::new(new(&self.dtype))));
        }
        let Some(CorePart::Storage(storage)) = self.core.last_mut() else {
            vortex_bail!("the Variant storage builder was just installed");
        };
        let Some(storage) = storage.as_any_mut().downcast_mut::<B>() else {
            vortex_bail!("the Variant storage builder has the type it was installed as");
        };
        append(storage)
    }

    /// Appends `n` copies of the Variant `scalar`.
    pub fn append_n_scalars(&mut self, scalar: &Scalar, n: usize) -> VortexResult<()> {
        self.check_dtype(scalar.dtype())?;
        if n == 0 {
            return Ok(());
        }
        let scalar = if scalar.dtype() == &self.dtype {
            scalar.clone()
        } else {
            scalar.cast(&self.dtype)?
        };
        if let Some(CorePart::Run(run, len)) = self.core.last_mut()
            && *run == scalar
        {
            *len += n;
            return Ok(());
        }
        self.push(CorePart::Run(scalar, n));
        Ok(())
    }

    /// Finishes the builder into a [`VariantArray`].
    pub fn finish_into_variant(&mut self) -> VortexResult<VariantArray> {
        let len = self.len();
        let shredded = self.finish_shredded(len)?;
        let mut chunks = std::mem::take(&mut self.core)
            .into_iter()
            .map(CorePart::finish)
            .collect::<VortexResult<Vec<_>>>()?;
        self.finished_len = 0;
        let core = match chunks.len() {
            1 => chunks.remove(0),
            // An empty Variant is an empty chunked array: there is no empty canonical storage.
            _ => ChunkedArray::try_new(chunks, self.dtype.clone())?.into_array(),
        };
        VariantArray::try_new(core, shredded)
    }

    /// Variants of the builder's nullability, and non-nullable Variants for a nullable builder,
    /// can be appended.
    fn check_dtype(&self, dtype: &DType) -> VortexResult<()> {
        vortex_ensure!(
            dtype.is_variant() && (self.dtype.is_nullable() || !dtype.is_nullable()),
            "cannot append {dtype} to a builder of {}",
            self.dtype
        );
        Ok(())
    }

    fn push(&mut self, part: CorePart) {
        if part.len() == 0 && !matches!(part, CorePart::Storage(_)) {
            return;
        }
        self.finished_len += self.core.last().map_or(0, CorePart::len);
        self.core.push(part);
    }

    /// Marks the rows up to `len` that have no shredded child.
    fn pad_shredded(&mut self, len: usize) {
        if len > self.shredded_len {
            self.shredded
                .push(ShreddedPart::Unshredded(len - self.shredded_len));
            self.shredded_len = len;
        }
    }

    fn finish_shredded(&mut self, len: usize) -> VortexResult<Option<ArrayRef>> {
        let Some(dtype) = self.shredded.iter().find_map(|part| match part {
            ShreddedPart::Array(array) => Some(array.dtype().clone()),
            ShreddedPart::Unshredded(_) => None,
        }) else {
            self.shredded.clear();
            self.shredded_len = 0;
            return Ok(None);
        };
        self.pad_shredded(len);
        let parts = std::mem::take(&mut self.shredded);
        self.shredded_len = 0;

        let nullable = parts.iter().any(|part| match part {
            ShreddedPart::Array(array) => array.dtype().is_nullable(),
            ShreddedPart::Unshredded(_) => true,
        });
        let dtype = if nullable { dtype.as_nullable() } else { dtype };
        let mut chunks = parts
            .into_iter()
            .map(|part| match part {
                ShreddedPart::Array(array) => {
                    vortex_ensure!(
                        array.dtype().eq_ignore_nullability(&dtype),
                        "cannot build a Variant from Variants shredded as {dtype} and as {}",
                        array.dtype()
                    );
                    if array.dtype() == &dtype {
                        Ok(array)
                    } else {
                        array.cast(dtype.clone())
                    }
                }
                ShreddedPart::Unshredded(len) => {
                    Ok(ConstantArray::new(Scalar::null(dtype.clone()), len).into_array())
                }
            })
            .collect::<VortexResult<Vec<_>>>()?;
        if chunks.len() == 1 {
            return Ok(chunks.pop());
        }
        Ok(Some(ChunkedArray::try_new(chunks, dtype)?.into_array()))
    }
}

impl ArrayBuilder for VariantBuilder {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.finished_len + self.core.last().map_or(0, CorePart::len)
    }

    fn append_zeros(&mut self, n: usize) {
        let zero = Scalar::zero_value(&self.dtype);
        self.append_n_scalars(&zero, n)
            .vortex_expect("the zero value of the builder's dtype can be appended");
    }

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        let null = Scalar::null(self.dtype.clone());
        self.append_n_scalars(&null, n)
            .vortex_expect("a null of the builder's dtype can be appended");
    }

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        self.append_n_scalars(scalar, 1)
    }

    fn reserve_exact(&mut self, _additional: usize) {}

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_variant()
            .vortex_expect("the appended Variants finish into one")
            .into_array()
    }

    fn finish_into_canonical(&mut self, _ctx: &mut ExecutionCtx) -> Canonical {
        Canonical::Variant(
            self.finish_into_variant()
                .vortex_expect("the appended Variants finish into one"),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use super::*;
    use crate::VortexSessionExecute;
    use crate::arrays::Chunked;
    use crate::arrays::Constant;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::chunked::ChunkedArrayExt;
    use crate::assert_arrays_eq;
    use crate::builders::builder_with_capacity_in;
    use crate::memory::BufferAllocatorRef;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn variant(value: i32) -> Scalar {
        Scalar::variant(Scalar::primitive(value, Nullability::NonNullable))
    }

    /// The `i32` each row holds, or `None` for a null row.
    fn values(array: &VariantArray) -> VortexResult<Vec<Option<i32>>> {
        let mut ctx = SESSION.create_execution_ctx();
        (0..array.len())
            .map(|idx| {
                let scalar = array.execute_scalar(idx, &mut ctx)?;
                Ok(scalar
                    .as_variant()
                    .value()
                    .and_then(|value| value.as_primitive().as_::<i32>()))
            })
            .collect()
    }

    #[test]
    fn dispatches_on_the_variant_dtype() {
        let builder = builder_with_capacity_in(
            &DType::Variant(Nullability::Nullable),
            4,
            BufferAllocatorRef::static_ref(),
        );
        assert!(builder.as_any().is::<VariantBuilder>());
    }

    /// Repeated scalars, nulls included, become one constant run each.
    #[test]
    fn appends_scalars_as_runs() -> VortexResult<()> {
        let mut builder = VariantBuilder::new(Nullability::Nullable);
        builder.append_scalar(&variant(1))?;
        builder.append_scalar(&variant(1))?;
        builder.append_nulls(2);
        builder.append_scalar(&variant(2))?;

        let array = builder.finish_into_variant()?;
        assert_eq!(values(&array)?, [Some(1), Some(1), None, None, Some(2)]);
        let core = array.core_storage().as_::<Chunked>();
        assert_eq!(core.nchunks(), 3);
        assert!(core.iter_chunks().all(|chunk| chunk.is::<Constant>()));
        Ok(())
    }

    /// Rows appended without a shredded child are null in it, so their values come from the
    /// core storage.
    #[test]
    fn pads_the_shredded_child() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let shredded = VariantArray::try_new(
            ConstantArray::new(variant(1), 2).into_array(),
            Some(PrimitiveArray::from_iter([10i32, 20]).into_array()),
        )?;
        let mut builder = VariantBuilder::new(Nullability::NonNullable);
        builder.append_scalar(&variant(3))?;
        builder.append_variant_array(&shredded, &mut ctx)?;
        builder.append_scalar(&variant(4))?;

        let array = builder.finish_into_variant()?;
        assert_eq!(values(&array)?, [Some(3), Some(10), Some(20), Some(4)]);
        let shredded = array
            .shredded()
            .ok_or_else(|| vortex_err!("expected a shredded child"))?;
        assert_arrays_eq!(
            shredded,
            PrimitiveArray::from_option_iter([None, Some(10i32), Some(20), None]),
            &mut ctx
        );
        Ok(())
    }

    #[test]
    fn rejects_nullable_variants_in_a_non_nullable_builder() {
        let mut builder = VariantBuilder::new(Nullability::NonNullable);
        let null = Scalar::null(DType::Variant(Nullability::Nullable));
        assert!(builder.append_scalar(&null).is_err());
    }

    #[test]
    fn finishes_empty() -> VortexResult<()> {
        let array = VariantBuilder::new(Nullability::Nullable).finish_into_variant()?;
        assert!(array.is_empty());
        assert!(array.shredded().is_none());
        Ok(())
    }
}
