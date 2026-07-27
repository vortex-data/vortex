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
use crate::ArrayView;
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
use crate::arrays::varbin::VarBinArraySlotsExt;
use crate::arrays::varbinview::VarBinViewArrayExt;
use crate::arrays::varbinview::build_views::BinaryView;
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

    /// Appends `n` non-null empty values.
    ///
    /// Every appended value has the same start and end offset, so no bytes are written.
    #[inline]
    pub fn append_n_empty(&mut self, n: usize) {
        self.offsets.push_n(self.offsets[self.offsets.len() - 1], n);
        self.validity.append_n(true, n);
    }

    /// Appends the same non-null value `n` times.
    #[inline]
    pub fn append_n_values(&mut self, value: impl AsRef<[u8]>, n: usize) {
        let slice = value.as_ref();
        if slice.is_empty() {
            return self.append_n_empty(n);
        }

        self.offsets.reserve(n);
        self.data.reserve(slice.len().saturating_mul(n));
        for _ in 0..n {
            self.append_value(slice);
        }
    }

    #[inline]
    pub fn append_values<P>(
        &mut self,
        values: &[u8],
        end_offsets: impl Iterator<Item = P>,
        validity: &Mask,
    ) -> VortexResult<()>
    where
        P: AsPrimitive<usize>,
    {
        let offsets_start = self.offsets.len();
        let data_start = self.data.len();
        let mut previous_end = data_start;
        let mut len = 0;

        for offset in end_offsets {
            let relative_end = offset.as_();
            let Some(end) = data_start.checked_add(relative_end) else {
                self.offsets.truncate(offsets_start);
                vortex_bail!("Byte offset overflow: {data_start} + {relative_end}");
            };
            if end < previous_end {
                self.offsets.truncate(offsets_start);
                vortex_bail!("End offsets must be monotonically increasing");
            }
            let Some(end_offset) = O::from(end) else {
                self.offsets.truncate(offsets_start);
                vortex_bail!(
                    "Byte offset {end} does not fit in {}",
                    std::any::type_name::<O>()
                );
            };
            self.offsets.push(end_offset);
            previous_end = end;
            len += 1;
        }

        if len != validity.len() {
            self.offsets.truncate(offsets_start);
            vortex_bail!(
                "End offset count {len} does not match validity length {}",
                validity.len()
            );
        }
        if previous_end - data_start != values.len() {
            self.offsets.truncate(offsets_start);
            vortex_bail!(
                "Final relative end offset {} does not match values length {}",
                previous_end - data_start,
                values.len()
            );
        }

        self.data.extend_from_slice(values);
        self.append_validity(validity);
        Ok(())
    }

    fn append_validity(&mut self, validity: &Mask) {
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
        assert_eq!(
            self.offsets.len() - 1,
            self.validity.len(),
            "The offset count must be one more than the validity length"
        );
        let offsets = PrimitiveArray::new(self.offsets.freeze(), Validity::NonNullable);
        let nulls = self.validity.freeze();

        let validity = Validity::from_bit_buffer(nulls, dtype.nullability());

        // The builder adds offsets in monotonically increasing order. Store this statistic to
        // prevent VarBinArray::validate from recomputing it after deserialization.
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
pub struct DynVarBinBuilder {
    dtype: DType,
    storage: DynOffsets,
}

enum DynOffsets {
    I32(VarBinBuilder<i32>),
    I64(VarBinBuilder<i64>),
}

impl DynVarBinBuilder {
    /// Creates a builder with 32-bit offsets, or 64-bit offsets when `large_offsets` is true.
    pub fn with_capacity(dtype: DType, large_offsets: bool, capacity: usize) -> Self {
        assert!(
            matches!(dtype, DType::Utf8(_) | DType::Binary(_)),
            "DynVarBinBuilder dtype must be Utf8 or Binary"
        );
        let storage = if large_offsets {
            DynOffsets::I64(VarBinBuilder::with_capacity(capacity))
        } else {
            DynOffsets::I32(VarBinBuilder::with_capacity(capacity))
        };
        Self { dtype, storage }
    }

    /// Appends decompressed values from one contiguous byte buffer.
    ///
    /// Each offset in `end_offsets` marks the end of one value relative to the start of `values`.
    pub fn append_values<P>(
        &mut self,
        values: &[u8],
        end_offsets: impl Iterator<Item = P>,
        validity: &Mask,
    ) -> VortexResult<()>
    where
        P: AsPrimitive<usize>,
    {
        match &mut self.storage {
            DynOffsets::I32(builder) => builder.append_values(values, end_offsets, validity),
            DynOffsets::I64(builder) => builder.append_values(values, end_offsets, validity),
        }
    }

    /// Appends the same non-null value `n` times.
    pub fn append_n_values(&mut self, value: impl AsRef<[u8]>, n: usize) {
        match &mut self.storage {
            DynOffsets::I32(builder) => builder.append_n_values(value, n),
            DynOffsets::I64(builder) => builder.append_n_values(value, n),
        }
    }

    /// Appends the same UTF-8 or binary scalar `n` times.
    pub fn append_scalar_repeated(&mut self, scalar: &Scalar, n: usize) -> VortexResult<()> {
        vortex_ensure!(
            scalar.dtype() == self.dtype(),
            "DynVarBinBuilder expected scalar with dtype {}, got {}",
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
            dtype => vortex_bail!("DynVarBinBuilder cannot append scalar of dtype {dtype}"),
        }
        Ok(())
    }

    /// Appends an existing [`VarBinArray`] without constructing views.
    pub fn append_varbin(
        &mut self,
        array: ArrayView<'_, VarBin>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let offsets = array.offsets().clone().execute::<PrimitiveArray>(ctx)?;
        let bytes: ByteBuffer = array.sliced_bytes();
        let validity = array
            .varbin_validity()
            .execute_mask(array.as_ref().len(), ctx)?;
        crate::match_each_integer_ptype!(offsets.ptype(), |P| {
            let offsets = offsets.as_slice::<P>();
            let first_offset: usize = offsets[0].as_();
            self.append_values(
                bytes.as_slice(),
                offsets
                    .iter()
                    .skip(1)
                    .map(|offset| AsPrimitive::<usize>::as_(*offset) - first_offset),
                &validity,
            )
        })?;
        Ok(())
    }

    /// Appends a [`VarBinViewArray`](crate::arrays::VarBinViewArray).
    pub fn append_varbinview(
        &mut self,
        array: ArrayView<'_, VarBinView>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let validity = array
            .varbinview_validity()
            .execute_mask(array.as_ref().len(), ctx)?;
        match &mut self.storage {
            DynOffsets::I32(builder) => append_varbinview_to(builder, array, &validity),
            DynOffsets::I64(builder) => append_varbinview_to(builder, array, &validity),
        }
        Ok(())
    }

    /// Finishes the current values as a [`VarBinArray`] and resets the builder.
    pub fn finish_into_varbin(&mut self) -> VarBinArray {
        match &mut self.storage {
            DynOffsets::I32(builder) => std::mem::take(builder).finish(self.dtype.clone()),
            DynOffsets::I64(builder) => std::mem::take(builder).finish(self.dtype.clone()),
        }
    }

    fn push_nulls(&mut self, n: usize) {
        match &mut self.storage {
            DynOffsets::I32(builder) => builder.append_n_nulls(n),
            DynOffsets::I64(builder) => builder.append_n_nulls(n),
        }
    }
}

/// Appends `array` to a [`VarBinBuilder`] with a statically known offset type.
///
/// [`DynVarBinBuilder`] resolves its offset width once before calling this, so the loops below
/// append through a monomorphized [`VarBinBuilder`] rather than re-reading [`DynOffsets`] per row.
fn append_varbinview_to<O: IntegerPType>(
    builder: &mut VarBinBuilder<O>,
    array: ArrayView<'_, VarBinView>,
    validity: &Mask,
) {
    // Read the views slice once. `bytes_at` would re-derive it and hand back a refcounted
    // `ByteBuffer` per row; `view_bytes` borrows the payload instead.
    let views = array.views();
    match validity {
        Mask::AllTrue(len) => {
            builder.reserve_exact(*len);
            for view in &views[..*len] {
                builder.append_value(view_bytes(&array, view));
            }
        }
        Mask::AllFalse(len) => builder.append_n_nulls(*len),
        Mask::Values(values) => {
            builder.reserve_exact(values.len());
            // Offsets and validity must be appended in row order, so this cannot use the
            // set-index fast paths: the null rows carry offsets too.
            for (view, is_valid) in views[..values.len()].iter().zip(values.bit_buffer().iter()) {
                if is_valid {
                    builder.append_value(view_bytes(&array, view));
                } else {
                    builder.append_null();
                }
            }
        }
    }
}

/// Borrows the bytes of a single view without constructing a [`ByteBuffer`].
#[inline]
fn view_bytes<'a>(array: &'a ArrayView<'_, VarBinView>, view: &'a BinaryView) -> &'a [u8] {
    if view.is_inlined() {
        view.as_inlined().value()
    } else {
        let view_ref = view.as_view();
        &array.buffer(view_ref.buffer_index as usize).as_slice()[view_ref.as_range()]
    }
}

impl ArrayBuilder for DynVarBinBuilder {
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
            DynOffsets::I32(builder) => builder.len(),
            DynOffsets::I64(builder) => builder.len(),
        }
    }

    fn append_zeros(&mut self, n: usize) {
        match &mut self.storage {
            DynOffsets::I32(builder) => builder.append_n_empty(n),
            DynOffsets::I64(builder) => builder.append_n_empty(n),
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
            DynOffsets::I32(builder) => builder.reserve_exact(additional),
            DynOffsets::I64(builder) => builder.reserve_exact(additional),
        }
    }

    unsafe fn set_validity_unchecked(&mut self, validity: Mask) {
        match &mut self.storage {
            DynOffsets::I32(builder) => builder.set_validity(validity),
            DynOffsets::I64(builder) => builder.set_validity(validity),
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
    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::varbin::VarBinArraySlotsExt;
    use crate::arrays::varbin::builder::DynVarBinBuilder;
    use crate::arrays::varbin::builder::VarBinBuilder;
    use crate::assert_arrays_eq;
    use crate::builders::ArrayBuilder;
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

    #[rstest]
    #[case(false)]
    #[case(true)]
    fn test_append_varbin_to_builder(#[case] large_offsets: bool) -> VortexResult<()> {
        let source = VarBinArray::from_iter(
            [
                Some("prefix"),
                Some("hello"),
                None,
                Some("world"),
                Some("suffix"),
            ],
            DType::Utf8(Nullable),
        )
        .into_array()
        .slice(1..4)?;
        let mut builder =
            DynVarBinBuilder::with_capacity(source.dtype().clone(), large_offsets, source.len());
        let mut ctx = array_session().create_execution_ctx();

        source.append_to_builder(&mut builder, &mut ctx)?;

        assert_arrays_eq!(builder.finish_into_varbin(), source, &mut ctx);
        Ok(())
    }

    #[test]
    fn append_values_offset_overflow_returns_error() {
        let mut builder = VarBinBuilder::<i8>::new();
        let values = [0u8; 128];

        let result = builder.append_values(&values, [values.len()].into_iter(), &Mask::new_true(1));

        assert!(result.is_err());
        assert_eq!(builder.offsets.len(), 1);
        assert!(builder.data.is_empty());
        assert_eq!(builder.validity.len(), 0);
    }

    #[test]
    #[should_panic(expected = "The offset count must be one more than the validity length")]
    fn finish_rejects_mismatched_validity() {
        let mut builder = VarBinBuilder::<i32>::new();
        builder.validity.append_true();
        drop(builder.finish(DType::Utf8(Nullable)));
    }

    #[rstest]
    #[case(false)]
    #[case(true)]
    fn test_array_builder_methods(#[case] large_offsets: bool) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        for validity in [
            Mask::new_true(3),
            Mask::new_false(3),
            Mask::from_iter([true, false, true]),
        ] {
            let mut builder =
                DynVarBinBuilder::with_capacity(DType::Utf8(Nullable), large_offsets, 0);
            assert!(builder.as_any().is::<DynVarBinBuilder>());
            builder.reserve_exact(3);
            builder.append_zero();
            builder.append_scalar(&Scalar::utf8("hello", Nullable))?;
            builder.append_null();
            assert_eq!(builder.len(), 3);
            builder.set_validity(validity.clone());

            let result = builder.finish_into_canonical(&mut ctx).into_array();
            assert_eq!(result.validity()?.execute_mask(3, &mut ctx)?, validity);
        }
        Ok(())
    }

    #[test]
    fn test_append_varbinview_validity_to_dyn_builder() -> VortexResult<()> {
        let all_null = VarBinViewArray::from_iter([None::<&str>, None], DType::Utf8(Nullable));
        let mixed =
            VarBinViewArray::from_iter([Some("hello"), None, Some("world")], DType::Utf8(Nullable));
        let expected = VarBinViewArray::from_iter(
            [None, None, Some("hello"), None, Some("world")],
            DType::Utf8(Nullable),
        );
        let mut builder =
            DynVarBinBuilder::with_capacity(expected.dtype().clone(), false, expected.len());
        let mut ctx = array_session().create_execution_ctx();

        all_null
            .into_array()
            .append_to_builder(&mut builder, &mut ctx)?;
        mixed
            .into_array()
            .append_to_builder(&mut builder, &mut ctx)?;

        assert_arrays_eq!(builder.finish_into_varbin(), expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case(false)]
    #[case(true)]
    fn append_zeros_appends_non_null_empty_values(#[case] large_offsets: bool) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = DynVarBinBuilder::with_capacity(DType::Utf8(Nullable), large_offsets, 0);

        builder.append_scalar(&Scalar::utf8("hello", Nullable))?;
        builder.append_zeros(3);

        assert_eq!(builder.len(), 4);
        let expected = VarBinArray::from_iter(
            [Some("hello"), Some(""), Some(""), Some("")],
            DType::Utf8(Nullable),
        );
        assert_arrays_eq!(builder.finish_into_varbin(), expected, &mut ctx);
        Ok(())
    }

    #[rstest]
    #[case(false)]
    #[case(true)]
    fn append_n_values_repeats_the_value(#[case] large_offsets: bool) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = DynVarBinBuilder::with_capacity(DType::Utf8(Nullable), large_offsets, 0);

        builder.append_n_values("ab", 2);
        builder.append_null();
        builder.append_n_values("", 2);

        let expected = VarBinArray::from_iter(
            [Some("ab"), Some("ab"), None, Some(""), Some("")],
            DType::Utf8(Nullable),
        );
        assert_arrays_eq!(builder.finish_into_varbin(), expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn append_n_empty_preserves_previous_offset() {
        let mut builder = VarBinBuilder::<i32>::new();
        builder.append_value(b"abc");
        builder.append_n_empty(2);
        let array = builder.finish(DType::Utf8(Nullable));

        assert_eq!(array.len(), 3);
        assert_eq!(array.bytes().len(), 3);
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
