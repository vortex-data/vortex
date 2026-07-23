// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use num_traits::AsPrimitive;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
#[cfg(debug_assertions)]
use crate::VortexSessionExecute;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBin;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinView;
use crate::arrays::varbin::VarBinArrayExt;
use crate::arrays::varbinview::VarBinViewArrayExt;
use crate::builders::ArrayBuilder;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
#[cfg(debug_assertions)]
use crate::legacy_session;
use crate::scalar::Scalar;
use crate::validity::Validity;
pub struct VarBinBuilder<O: IntegerPType> {
    offsets: BufferMut<O>,
    data: BufferMut<u8>,
    validity: BitBufferMut,
}

impl<O: IntegerPType> Default for VarBinBuilder<O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<O: IntegerPType> VarBinBuilder<O> {
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(len: usize) -> Self {
        let mut offsets = BufferMut::with_capacity(len + 1);
        offsets.push(O::zero());
        Self {
            offsets,
            data: BufferMut::empty(),
            validity: BitBufferMut::with_capacity(len),
        }
    }

    #[inline]
    pub fn append(&mut self, value: Option<&[u8]>) {
        match value {
            Some(v) => self.append_value(v),
            None => self.append_null(),
        }
    }

    #[inline]
    pub fn append_value(&mut self, value: impl AsRef<[u8]>) {
        let slice = value.as_ref();
        self.offsets
            .push(O::from(self.data.len() + slice.len()).unwrap_or_else(|| {
                vortex_panic!(
                    "Failed to convert sum of {} and {} to offset of type {}",
                    self.data.len(),
                    slice.len(),
                    std::any::type_name::<O>()
                )
            }));
        self.data.extend_from_slice(slice);
        self.validity.append_true();
    }

    #[inline]
    pub fn append_null(&mut self) {
        self.offsets.push(self.offsets[self.offsets.len() - 1]);
        self.validity.append_false();
    }

    #[inline]
    pub fn append_n_nulls(&mut self, n: usize) {
        self.offsets.push_n(self.offsets[self.offsets.len() - 1], n);
        self.validity.append_n(false, n);
    }

    #[inline]
    pub fn append_values(&mut self, values: &[u8], end_offsets: impl Iterator<Item = O>, num: usize)
    where
        O: 'static,
        usize: AsPrimitive<O>,
    {
        self.offsets
            .extend(end_offsets.map(|offset| offset + self.data.len().as_()));
        self.data.extend_from_slice(values);
        self.validity.append_n(true, num);
    }

    fn append_values_with_lengths(
        &mut self,
        values: &[u8],
        lengths: impl Iterator<Item = usize>,
        validity: &Mask,
    ) {
        let mut end = self.data.len();
        let mut len = 0;
        for value_len in lengths {
            end += value_len;
            self.offsets.push(O::from(end).unwrap_or_else(|| {
                vortex_panic!(
                    "Failed to convert byte offset {end} to {}",
                    std::any::type_name::<O>()
                )
            }));
            len += 1;
        }
        debug_assert_eq!(len, validity.len());
        debug_assert_eq!(end - self.data.len(), values.len());
        self.data.extend_from_slice(values);
        match validity {
            Mask::AllTrue(len) => self.validity.append_n(true, *len),
            Mask::AllFalse(len) => self.validity.append_n(false, *len),
            Mask::Values(values) => self.validity.append_buffer(values.bit_buffer()),
        }
    }

    fn len(&self) -> usize {
        self.validity.len()
    }

    fn reserve_exact(&mut self, additional: usize) {
        self.offsets.reserve(additional);
        self.validity.reserve(additional);
    }

    fn set_validity(&mut self, validity: Mask) {
        self.validity = match validity {
            Mask::AllTrue(len) => BitBufferMut::new_set(len),
            Mask::AllFalse(len) => BitBufferMut::new_unset(len),
            values @ Mask::Values(_) => values
                .into_bit_buffer()
                .try_into_mut()
                .unwrap_or_else(|buffer| BitBufferMut::copy_from(&buffer)),
        };
    }

    #[allow(clippy::disallowed_methods)]
    pub fn finish(self, dtype: DType) -> VarBinArray {
        let offsets = PrimitiveArray::new(self.offsets.freeze(), Validity::NonNullable);
        let nulls = self.validity.freeze();

        let validity = Validity::from_bit_buffer(nulls, dtype.nullability());

        // The builder guarantees offsets are monotonically increasing, so we can set
        // this stat eagerly. This avoids an O(n) recomputation when the array is
        // deserialized and VarBinArray::validate checks sortedness.
        #[cfg(debug_assertions)]
        {
            let offsets_are_sorted = offsets
                .statistics()
                .compute_is_sorted(&mut legacy_session().create_execution_ctx())
                .unwrap_or(false);
            debug_assert!(offsets_are_sorted, "VarBinBuilder offsets must be sorted");
        }
        offsets
            .statistics()
            .set(Stat::IsSorted, Precision::Exact(true.into()));

        // SAFETY: The builder maintains all invariants:
        // - Offsets are monotonically increasing starting from 0 (guaranteed by builder logic).
        // - Bytes buffer contains exactly the data referenced by offsets.
        // - Validity matches the dtype nullability.
        // - UTF-8 validity is ensured by the caller when using DType::Utf8.
        unsafe {
            VarBinArray::new_unchecked(offsets.into_array(), self.data.freeze(), dtype, validity)
        }
    }
}

/// Builder for UTF-8 and binary [`VarBinArray`] values.
///
/// Unlike [`crate::builders::VarBinViewBuilder`], this builder stores 32-bit or 64-bit offsets.
/// Encodings can decode into this builder without first creating a
/// [`VarBinViewArray`](crate::arrays::VarBinViewArray).
pub struct VarBinBufferBuilder {
    dtype: DType,
    storage: TypedBuilder,
}

enum TypedBuilder {
    I32(VarBinBuilder<i32>),
    I64(VarBinBuilder<i64>),
}

impl VarBinBufferBuilder {
    /// Creates a builder with 32-bit offsets, or 64-bit offsets when `large_offsets` is true.
    pub fn with_capacity(dtype: DType, large_offsets: bool, capacity: usize) -> Self {
        assert!(matches!(dtype, DType::Utf8(_) | DType::Binary(_)));
        let storage = if large_offsets {
            TypedBuilder::I64(VarBinBuilder::with_capacity(capacity))
        } else {
            TypedBuilder::I32(VarBinBuilder::with_capacity(capacity))
        };
        Self { dtype, storage }
    }

    /// Returns whether this builder uses 64-bit offsets.
    pub fn has_large_offsets(&self) -> bool {
        matches!(self.storage, TypedBuilder::I64(_))
    }

    /// Appends decompressed values represented by one contiguous byte buffer and per-row lengths.
    pub fn append_values(
        &mut self,
        values: &[u8],
        lengths: impl Iterator<Item = usize>,
        validity: &Mask,
    ) {
        match &mut self.storage {
            TypedBuilder::I32(builder) => {
                builder.append_values_with_lengths(values, lengths, validity)
            }
            TypedBuilder::I64(builder) => {
                builder.append_values_with_lengths(values, lengths, validity)
            }
        }
    }

    /// Appends the same non-null value `n` times.
    pub fn append_n_values(&mut self, value: impl AsRef<[u8]>, n: usize) {
        let value = value.as_ref();
        for _ in 0..n {
            self.append_value(value);
        }
    }

    /// Appends the same UTF-8 or binary scalar `n` times.
    pub fn append_scalar_repeated(&mut self, scalar: &Scalar, n: usize) -> VortexResult<()> {
        vortex_ensure!(
            scalar.dtype() == self.dtype(),
            "VarBinBufferBuilder expected scalar with dtype {}, got {}",
            self.dtype(),
            scalar.dtype()
        );
        match self.dtype() {
            DType::Utf8(_) => match scalar.as_utf8().value() {
                Some(value) => self.append_n_values(value, n),
                None => self.push_nulls(n),
            },
            DType::Binary(_) => match scalar.as_binary().value() {
                Some(value) => self.append_n_values(value, n),
                None => self.push_nulls(n),
            },
            dtype => vortex_bail!("VarBinBufferBuilder cannot append scalar of dtype {dtype}"),
        }
        Ok(())
    }

    /// Appends an existing [`VarBinArray`] without constructing views.
    pub fn append_varbin(
        &mut self,
        array: crate::ArrayView<'_, VarBin>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let offsets = array.offsets().clone().execute::<PrimitiveArray>(ctx)?;
        let bytes: ByteBuffer = array.sliced_bytes();
        let validity = array
            .varbin_validity()
            .execute_mask(array.as_ref().len(), ctx)?;
        crate::match_each_integer_ptype!(offsets.ptype(), |P| {
            let offsets = offsets.as_slice::<P>();
            self.append_values(
                bytes.as_slice(),
                offsets.windows(2).map(|window| {
                    let start: usize = window[0].as_();
                    let end: usize = window[1].as_();
                    end - start
                }),
                &validity,
            );
        });
        Ok(())
    }

    /// Appends a [`VarBinViewArray`](crate::arrays::VarBinViewArray).
    pub fn append_varbinview(
        &mut self,
        array: crate::ArrayView<'_, VarBinView>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let validity = array
            .varbinview_validity()
            .execute_mask(array.as_ref().len(), ctx)?;
        match &validity {
            Mask::AllTrue(_) => {
                for index in 0..array.as_ref().len() {
                    self.append_value(array.bytes_at(index));
                }
            }
            Mask::AllFalse(len) => self.push_nulls(*len),
            Mask::Values(values) => {
                for (index, is_valid) in values.bit_buffer().iter().enumerate() {
                    if is_valid {
                        self.append_value(array.bytes_at(index));
                    } else {
                        self.push_null();
                    }
                }
            }
        }
        Ok(())
    }

    /// Finishes the current values as a [`VarBinArray`] and resets the builder.
    pub fn finish_into_varbin(&mut self) -> VarBinArray {
        match &mut self.storage {
            TypedBuilder::I32(builder) => std::mem::take(builder).finish(self.dtype.clone()),
            TypedBuilder::I64(builder) => std::mem::take(builder).finish(self.dtype.clone()),
        }
    }

    fn append_value(&mut self, value: impl AsRef<[u8]>) {
        match &mut self.storage {
            TypedBuilder::I32(builder) => builder.append_value(value),
            TypedBuilder::I64(builder) => builder.append_value(value),
        }
    }

    fn push_null(&mut self) {
        match &mut self.storage {
            TypedBuilder::I32(builder) => builder.append_null(),
            TypedBuilder::I64(builder) => builder.append_null(),
        }
    }

    fn push_nulls(&mut self, n: usize) {
        match &mut self.storage {
            TypedBuilder::I32(builder) => builder.append_n_nulls(n),
            TypedBuilder::I64(builder) => builder.append_n_nulls(n),
        }
    }
}

impl ArrayBuilder for VarBinBufferBuilder {
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
        match &self.storage {
            TypedBuilder::I32(builder) => builder.len(),
            TypedBuilder::I64(builder) => builder.len(),
        }
    }

    fn append_zeros(&mut self, n: usize) {
        for _ in 0..n {
            self.append_value([]);
        }
    }

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        self.push_nulls(n);
    }

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        self.append_scalar_repeated(scalar, 1)
    }

    fn reserve_exact(&mut self, additional: usize) {
        match &mut self.storage {
            TypedBuilder::I32(builder) => builder.reserve_exact(additional),
            TypedBuilder::I64(builder) => builder.reserve_exact(additional),
        }
    }

    unsafe fn set_validity_unchecked(&mut self, validity: Mask) {
        match &mut self.storage {
            TypedBuilder::I32(builder) => builder.set_validity(validity),
            TypedBuilder::I64(builder) => builder.set_validity(validity),
        }
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_varbin().into_array()
    }

    fn finish_into_canonical(&mut self, ctx: &mut ExecutionCtx) -> Canonical {
        self.finish()
            .execute::<Canonical>(ctx)
            .vortex_expect("varbin buffer builder should canonicalize")
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::varbin::VarBinArrayExt;
    use crate::arrays::varbin::builder::VarBinBuilder;
    use crate::dtype::DType;
    use crate::dtype::Nullability::Nullable;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::expr::stats::StatsProviderExt;
    use crate::scalar::Scalar;

    #[test]
    fn test_builder() {
        let mut builder = VarBinBuilder::<i32>::with_capacity(0);
        builder.append(Some(b"hello"));
        builder.append(None);
        builder.append(Some(b"world"));
        let array = builder.finish(DType::Utf8(Nullable));

        assert_eq!(array.len(), 3);
        assert_eq!(array.dtype().nullability(), Nullable);
        assert_eq!(
            array
                .execute_scalar(0, &mut array_session().create_execution_ctx())
                .unwrap(),
            Scalar::utf8("hello".to_string(), Nullable)
        );
        assert!(
            array
                .execute_scalar(1, &mut array_session().create_execution_ctx())
                .unwrap()
                .is_null()
        );
    }

    #[test]
    fn offsets_have_is_sorted_stat() -> VortexResult<()> {
        let mut builder = VarBinBuilder::<i32>::with_capacity(0);
        builder.append_value(b"aaa");
        builder.append_null();
        builder.append_value(b"bbb");
        let array = builder.finish(DType::Utf8(Nullable));

        let is_sorted = array
            .offsets()
            .statistics()
            .with_typed_stats_set(|s| s.get_as::<bool>(Stat::IsSorted));
        assert_eq!(is_sorted, Precision::Exact(true));
        Ok(())
    }

    #[test]
    fn empty_builder_offsets_have_is_sorted_stat() -> VortexResult<()> {
        let builder = VarBinBuilder::<i32>::new();
        let array = builder.finish(DType::Utf8(Nullable));

        let is_sorted = array
            .offsets()
            .statistics()
            .with_typed_stats_set(|s| s.get_as::<bool>(Stat::IsSorted));
        assert_eq!(is_sorted, Precision::Exact(true));
        Ok(())
    }
}
