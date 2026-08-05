// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::mem::MaybeUninit;

use num_traits::AsPrimitive;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;
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
use crate::builders::ArrayBuilder;
use crate::dtype::DType;
use crate::dtype::OffsetBuilderPType;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
#[cfg(debug_assertions)]
use crate::legacy_session;
use crate::match_each_integer_ptype;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// Builder for [`VarBinArray`] values with `O`-typed offsets.
///
/// This is the offset-based counterpart to
/// [`VarBinViewBuilder`](crate::builders::VarBinViewBuilder): values are laid out contiguously in
/// a single byte buffer described by a monotonically increasing offsets buffer. An `i32` or `i64`
/// builder can therefore be handed straight to Arrow as a `Utf8`/`LargeUtf8`/`Binary`/
/// `LargeBinary` array without re-laying out the bytes.
///
/// Encodings that can decode into this layout should specialize
/// [`append_to_builder`](crate::vtable::VTable::append_to_builder) with
/// [`match_each_varbin_builder!`](crate::match_each_varbin_builder), which recovers the concrete
/// offset type from a `&mut dyn ArrayBuilder` so the decode loop is monomorphized over it.
pub struct VarBinBuilder<O: OffsetBuilderPType> {
    dtype: DType,
    offsets: BufferMut<O>,
    data: ByteBufferMut,
    validity: BitBufferMut,
}

impl<O: OffsetBuilderPType> VarBinBuilder<O> {
    /// Creates an empty builder for `dtype`.
    pub fn new(dtype: DType) -> Self {
        Self::with_capacity(dtype, 0)
    }

    /// Creates a builder for `dtype` with room for `capacity` values.
    pub fn with_capacity(dtype: DType, capacity: usize) -> Self {
        assert!(
            matches!(dtype, DType::Utf8(_) | DType::Binary(_)),
            "VarBinBuilder dtype must be Utf8 or Binary, got {dtype}"
        );
        let mut offsets = BufferMut::with_capacity(capacity + 1);
        offsets.push(O::zero());
        Self {
            dtype,
            offsets,
            data: BufferMut::empty(),
            validity: BitBufferMut::with_capacity(capacity),
        }
    }

    /// Creates a builder for `dtype` with room for `capacity` values totalling `bytes` bytes.
    pub fn with_capacity_bytes(dtype: DType, capacity: usize, bytes: usize) -> Self {
        let mut builder = Self::with_capacity(dtype, capacity);
        builder.reserve_data(bytes);
        builder
    }

    /// Reserves room for `additional` value bytes.
    ///
    /// [`reserve_exact`](ArrayBuilder::reserve_exact) takes a row count and so cannot size the
    /// value bytes; callers that know the byte total should use this to size the buffer once
    /// instead of letting it grow.
    pub fn reserve_data(&mut self, additional: usize) {
        self.data.reserve(additional);
    }

    /// Appends one value, or a null when it is `None`.
    ///
    /// # Panics
    ///
    /// Panics if the resulting end offset does not fit in `O`. See
    /// [`append_value`](Self::append_value).
    #[inline]
    pub fn append(&mut self, value: Option<&[u8]>) {
        match value {
            Some(v) => self.append_value(v),
            None => self.push_null(),
        }
    }

    /// Appends one non-null value.
    ///
    /// # Panics
    ///
    /// Panics if the resulting end offset does not fit in `O`. The bulk appends report that as an
    /// error instead; use [`append_n_values`](Self::append_n_values) for a fallible single append,
    /// or an `i64` builder for byte totals past `i32::MAX`.
    #[inline]
    pub fn append_value(&mut self, value: impl AsRef<[u8]>) {
        self.push_value(value.as_ref());
        self.validity.append_true();
    }

    /// Appends the same non-null value `n` times.
    ///
    /// # Errors
    ///
    /// Returns an error, leaving the builder unchanged, if the resulting end offsets do not fit
    /// in `O`.
    pub fn append_n_values(&mut self, value: impl AsRef<[u8]>, n: usize) -> VortexResult<()> {
        let value = value.as_ref();
        let Some(num_bytes) = value.len().checked_mul(n) else {
            vortex_bail!("Byte count overflow: {} values of {} bytes", n, value.len());
        };
        // Checking the total up front is what keeps `push_value` below from panicking.
        self.check_offset_limit(self.data.len(), num_bytes)?;
        self.offsets.reserve(n);
        self.data.reserve(num_bytes);
        for _ in 0..n {
            self.push_value(value);
        }
        self.validity.append_n(true, n);
        Ok(())
    }

    /// Appends a null value.
    ///
    /// Unlike [`append_null`](ArrayBuilder::append_null) this does not check that the builder is
    /// nullable; the offsets and validity stay consistent either way, and a non-nullable `dtype`
    /// discards the validity bits on [`finish_into_varbin`](Self::finish_into_varbin).
    #[inline]
    pub fn push_null(&mut self) {
        self.push_nulls(1)
    }

    /// Appends `n` null values. See [`push_null`](Self::push_null).
    #[inline]
    pub fn push_nulls(&mut self, n: usize) {
        self.offsets.push_n(self.last_offset(), n);
        self.validity.append_n(false, n);
    }

    /// Appends the same UTF-8 or binary scalar `n` times.
    ///
    /// # Errors
    ///
    /// Returns an error if `scalar` has a different dtype than the builder, or if the resulting
    /// end offsets do not fit in `O`.
    pub fn append_scalar_repeated(&mut self, scalar: &Scalar, n: usize) -> VortexResult<()> {
        vortex_ensure!(
            scalar.dtype() == &self.dtype,
            "VarBinBuilder expected scalar with dtype {}, got {}",
            self.dtype,
            scalar.dtype()
        );
        match &self.dtype {
            DType::Utf8(_) => match scalar.as_utf8().value() {
                Some(value) => self.append_n_values(value, n)?,
                None => self.push_nulls(n),
            },
            DType::Binary(_) => match scalar.as_binary().value() {
                Some(value) => self.append_n_values(value, n)?,
                None => self.push_nulls(n),
            },
            dtype => vortex_bail!("VarBinBuilder cannot append scalar of dtype {dtype}"),
        }
        Ok(())
    }

    /// Appends values from one contiguous byte buffer described by relative end offsets.
    ///
    /// Each entry of `end_offsets` marks the end of one value relative to the start of `values`,
    /// and there must be exactly one per entry in `validity`.
    #[inline]
    pub fn append_values<P>(
        &mut self,
        values: &[u8],
        end_offsets: impl Iterator<Item = P>,
        validity: &Mask,
    ) -> VortexResult<()>
    where
        P: AsPrimitive<usize>,
        usize: AsPrimitive<O>,
    {
        // Offsets are committed first: they are the only fallible part, and failing before the
        // bytes are appended leaves the builder untouched.
        self.extend_offsets(values.len(), validity.len(), end_offsets)?;
        self.data.extend_from_slice(values);
        self.append_validity(validity);
        Ok(())
    }

    /// Appends values whose bytes `decode` writes straight into the builder's byte storage.
    ///
    /// `decode` is handed at least `num_bytes + slack` bytes of uninitialized storage and returns
    /// the number of bytes it initialized, which must be exactly `num_bytes`. Each entry of
    /// `lengths` is the byte length of one value — including nulls, which are zero-length — so
    /// there must be one per entry in `validity`.
    ///
    /// `slack` is spare headroom past the values themselves. It exists so that a decoder which
    /// stores in wide fixed-size chunks can keep using them through the final value instead of
    /// dropping into a byte-at-a-time tail; a decoder must still never write past the slice it is
    /// handed, so `0` is correct for one that is already exact.
    ///
    /// Decoding in place saves staging the decoded heap in a temporary buffer and copying it in.
    ///
    /// `decode` is taken as a `dyn` reference so a caller that resolves the `lengths` type with
    /// [`match_each_integer_ptype!`](crate::match_each_integer_ptype) can build the closure once
    /// outside that match rather than inlining a whole decoder into each of its arms.
    ///
    /// # Errors
    ///
    /// Returns an error, leaving the builder unchanged, if `num_bytes` would push an offset past
    /// what `O` can hold — checked before `decode` runs — or if `decode` reports a different byte
    /// count than `num_bytes`, or if `lengths` does not describe those bytes.
    ///
    /// # Safety
    ///
    /// `decode` must initialize the first `n` bytes of the slice it is passed, where `n` is the
    /// value it returns. Those bytes are published as initialized without ever being read first,
    /// so a `decode` that over-reports leaves the builder holding uninitialized memory.
    pub unsafe fn append_decoded<P>(
        &mut self,
        num_bytes: usize,
        slack: usize,
        lengths: &[P],
        validity: &Mask,
        decode: &mut dyn FnMut(&mut [MaybeUninit<u8>]) -> VortexResult<usize>,
    ) -> VortexResult<()>
    where
        P: AsPrimitive<usize>,
        usize: AsPrimitive<O>,
    {
        let Some(capacity) = num_bytes.checked_add(slack) else {
            vortex_bail!("Decoded size overflow: {num_bytes} + {slack}");
        };
        // Checked before decoding: an `i32` builder that the decoded bytes would overflow should
        // not pay for the decompression first.
        self.check_offset_limit(self.data.len(), num_bytes)?;
        self.data.reserve(capacity);

        let data_len = self.data.len();
        let written = decode(self.data.spare_capacity_mut())?;
        vortex_ensure!(
            written == num_bytes,
            "Decoded {written} bytes, expected {num_bytes}"
        );

        // The decoded bytes live in spare capacity until `set_len` below, so an invalid `lengths`
        // still leaves the builder unchanged.
        self.extend_offsets(num_bytes, validity.len(), prefix_sums(lengths))?;

        // SAFETY: `decode` reported initializing `written` spare bytes, and the caller guarantees
        // that report is accurate; `written == num_bytes` was checked above.
        unsafe { self.data.set_len(data_len + num_bytes) };
        self.append_validity(validity);
        Ok(())
    }

    /// Appends `validity.len()` values, taking each valid row's bytes from `value`.
    ///
    /// `value` is called once per valid row in ascending row order and the slices it returns must
    /// total `num_bytes` bytes. Null rows consume no bytes and repeat the preceding offset.
    ///
    /// The valid rows are walked a `u64` word at a time rather than pulled one at a time through
    /// an iterator, and both buffers are sized once up front, so the copy loop costs one offset
    /// store plus one `memcpy` per value. A caller whose values only exist as a stream can still
    /// use this by pulling from the stream inside `value`: the rows are visited in ascending
    /// order.
    pub fn append_valid_slices<'a, F>(
        &mut self,
        num_bytes: usize,
        validity: &Mask,
        value: F,
    ) -> VortexResult<()>
    where
        F: FnMut(usize) -> &'a [u8],
        usize: AsPrimitive<O>,
    {
        let data_start = self.data.len();
        match self.gather_valid_slices(num_bytes, validity, value) {
            Ok(()) => {
                self.append_validity(validity);
                Ok(())
            }
            Err(error) => {
                // The offsets are never committed on failure, so only the copied bytes need to go.
                self.data.truncate(data_start);
                Err(error)
            }
        }
    }

    /// Appends a [`VarBinArray`], reusing its offsets instead of walking its values.
    pub fn append_varbin(
        &mut self,
        array: ArrayView<'_, VarBin>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()>
    where
        usize: AsPrimitive<O>,
    {
        let offsets = array.offsets().clone().execute::<PrimitiveArray>(ctx)?;
        let bytes: ByteBuffer = array.sliced_bytes();
        let validity = array
            .varbin_validity()
            .execute_mask(array.as_ref().len(), ctx)?;
        match_each_integer_ptype!(offsets.ptype(), |P| {
            let offsets = offsets.as_slice::<P>();
            let first: usize = offsets[0].as_();
            self.append_values(
                bytes.as_slice(),
                // Wrapping keeps a corrupt offsets child from panicking here; `append_values`
                // rejects the resulting non-monotonic offsets.
                offsets[1..]
                    .iter()
                    .map(|offset| AsPrimitive::<usize>::as_(*offset).wrapping_sub(first)),
                &validity,
            )
        })
    }

    /// Appends a [`VarBinViewArray`](crate::arrays::VarBinViewArray), compacting its values.
    pub fn append_varbinview(
        &mut self,
        array: ArrayView<'_, VarBinView>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()>
    where
        usize: AsPrimitive<O>,
    {
        let len = array.as_ref().len();
        let validity = array.varbinview_validity().execute_mask(len, ctx)?;

        // Resolve the views slice and the data buffers once. Reading them per row costs a buffer
        // handle clone, and the byte total needs only the fixed-width view headers.
        let views = array.views();
        let buffers = array
            .data_buffers()
            .iter()
            .map(|buffer| buffer.as_host().as_slice())
            .collect::<Vec<_>>();

        let num_bytes = match validity.bit_buffer() {
            AllOr::All => views.iter().map(|view| view.len() as usize).sum(),
            AllOr::None => 0,
            AllOr::Some(bits) => {
                let mut total = 0;
                bits.for_each_set_index(|index| total += views[index].len() as usize);
                total
            }
        };

        self.append_valid_slices(num_bytes, &validity, |index| views[index].bytes(&buffers))
    }

    /// Finishes the appended values into a [`VarBinArray`] and resets the builder.
    #[allow(clippy::disallowed_methods)]
    pub fn finish_into_varbin(&mut self) -> VarBinArray {
        assert_eq!(
            self.offsets.len() - 1,
            self.validity.len(),
            "The offset count must be one more than the validity length"
        );

        let mut fresh_offsets = BufferMut::with_capacity(1);
        fresh_offsets.push(O::zero());
        let offsets = PrimitiveArray::new(
            std::mem::replace(&mut self.offsets, fresh_offsets).freeze(),
            Validity::NonNullable,
        );
        let data = std::mem::replace(&mut self.data, BufferMut::empty());
        let nulls = std::mem::replace(&mut self.validity, BitBufferMut::empty()).freeze();

        let validity = Validity::from_bit_buffer(nulls, self.dtype.nullability());

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
            VarBinArray::new_unchecked(
                offsets.into_array(),
                data.freeze(),
                self.dtype.clone(),
                validity,
            )
        }
    }

    #[inline]
    fn last_offset(&self) -> O {
        self.offsets[self.offsets.len() - 1]
    }

    /// Appends `value`'s bytes and its end offset, leaving the validity bits to the caller.
    #[inline]
    fn push_value(&mut self, value: &[u8]) {
        self.offsets
            .push(O::from(self.data.len() + value.len()).unwrap_or_else(|| {
                vortex_panic!(
                    "Failed to convert sum of {} and {} to offset of type {}",
                    self.data.len(),
                    value.len(),
                    std::any::type_name::<O>()
                )
            }));
        self.data.extend_from_slice(value);
    }

    fn append_validity(&mut self, validity: &Mask) {
        match validity {
            Mask::AllTrue(len) => self.validity.append_n(true, *len),
            Mask::AllFalse(len) => self.validity.append_n(false, *len),
            Mask::Values(values) => self.validity.append_buffer(values.bit_buffer()),
        }
    }

    fn replace_validity(&mut self, validity: Mask) {
        self.validity = match validity {
            Mask::AllTrue(len) => BitBufferMut::new_set(len),
            Mask::AllFalse(len) => BitBufferMut::new_unset(len),
            values @ Mask::Values(_) => values
                .into_bit_buffer()
                .try_into_mut()
                .unwrap_or_else(|buffer| BitBufferMut::copy_from(&buffer)),
        };
    }

    /// Appends `count` end offsets derived from `end_offsets`, shifted past the current data end.
    ///
    /// `end_offsets` must be monotonically non-decreasing and end at exactly `num_bytes`. Offsets
    /// are written through the reserved spare capacity and committed with a single `set_len`, so a
    /// rejected input leaves the buffer untouched and the caller can propagate the error.
    fn extend_offsets<P>(
        &mut self,
        num_bytes: usize,
        count: usize,
        end_offsets: impl Iterator<Item = P>,
    ) -> VortexResult<()>
    where
        P: AsPrimitive<usize>,
        usize: AsPrimitive<O>,
    {
        let data_start = self.data.len();
        let offsets_len = self.offsets.len();
        self.check_offset_limit(data_start, num_bytes)?;
        self.offsets.reserve(count);

        // Writing into the spare capacity keeps the output cursor in a register: `push` rewrites
        // the buffer length on every value, which the optimizer cannot hoist out of the loop.
        let spare = &mut self.offsets.spare_capacity_mut()[..count];
        let mut end_offsets = end_offsets;
        let mut previous = 0usize;
        for slot in spare.iter_mut() {
            let Some(end) = end_offsets.next() else {
                vortex_bail!("End offset count is less than the validity length {count}");
            };
            let end = end.as_();
            vortex_ensure!(
                end >= previous && end <= num_bytes,
                "End offsets must be monotonically increasing within {num_bytes} bytes, \
                 got {end} after {previous}"
            );
            slot.write((data_start + end).as_());
            previous = end;
        }
        vortex_ensure!(
            end_offsets.next().is_none(),
            "End offset count exceeds the validity length {count}"
        );
        vortex_ensure!(
            previous == num_bytes,
            "Final end offset {previous} does not match the value byte count {num_bytes}"
        );

        // SAFETY: the loop initialized the first `count` spare slots.
        unsafe { self.offsets.set_len(offsets_len + count) };
        Ok(())
    }

    /// Copies each valid row's bytes from `value` into the byte storage and records the offsets.
    /// See [`append_valid_slices`](Self::append_valid_slices); the validity bits are the caller's
    /// job so that a failure here can be unwound.
    fn gather_valid_slices<'a, F>(
        &mut self,
        num_bytes: usize,
        validity: &Mask,
        mut value: F,
    ) -> VortexResult<()>
    where
        F: FnMut(usize) -> &'a [u8],
        usize: AsPrimitive<O>,
    {
        let count = validity.len();
        let data_start = self.data.len();
        let offsets_len = self.offsets.len();
        self.check_offset_limit(data_start, num_bytes)?;
        self.offsets.reserve(count);
        self.data.reserve(num_bytes);

        // Disjoint field borrows: the offsets spare capacity stays valid while the byte buffer
        // grows, since the two are separate allocations.
        let Self { offsets, data, .. } = self;
        let spare = &mut offsets.spare_capacity_mut()[..count];

        match validity.bit_buffer() {
            AllOr::All => {
                for (row, slot) in spare.iter_mut().enumerate() {
                    data.extend_from_slice(value(row));
                    slot.write(data.len().as_());
                }
            }
            AllOr::None => {
                spare.fill(MaybeUninit::new(data_start.as_()));
            }
            AllOr::Some(bits) => {
                let mut row = 0;
                bits.for_each_set_index(|index| {
                    // Null rows between the previous valid row and this one repeat its end offset.
                    spare[row..index].fill(MaybeUninit::new(data.len().as_()));
                    data.extend_from_slice(value(index));
                    spare[index].write(data.len().as_());
                    row = index + 1;
                });
                spare[row..].fill(MaybeUninit::new(data.len().as_()));
            }
        }

        // A caller whose slices overrun `num_bytes` only grows the byte buffer past the reservation,
        // so the overrun is caught here rather than per value. The offsets are still uncommitted, so
        // rejecting it now leaves them untouched.
        vortex_ensure!(
            data.len() == data_start + num_bytes,
            "Value slices total {} bytes, expected {num_bytes}",
            data.len() - data_start
        );

        // SAFETY: every branch above initialized all `count` spare slots.
        unsafe { self.offsets.set_len(offsets_len + count) };
        Ok(())
    }

    /// Checks that an offset past `num_bytes` more bytes is representable as an `O`.
    fn check_offset_limit(&self, data_start: usize, num_bytes: usize) -> VortexResult<()> {
        let Some(limit) = data_start.checked_add(num_bytes) else {
            vortex_bail!("Byte offset overflow: {data_start} + {num_bytes}");
        };
        vortex_ensure!(
            u64::try_from(limit).is_ok_and(|limit| limit <= O::max_value_as_u64()),
            "Byte offset {limit} does not fit in {}",
            std::any::type_name::<O>()
        );
        Ok(())
    }
}

impl<O: OffsetBuilderPType> ArrayBuilder for VarBinBuilder<O> {
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
        self.validity.len()
    }

    fn append_zeros(&mut self, n: usize) {
        self.offsets.push_n(self.last_offset(), n);
        self.validity.append_n(true, n);
    }

    unsafe fn append_nulls_unchecked(&mut self, n: usize) {
        self.push_nulls(n);
    }

    fn append_scalar(&mut self, scalar: &Scalar) -> VortexResult<()> {
        self.append_scalar_repeated(scalar, 1)
    }

    fn reserve_exact(&mut self, additional: usize) {
        self.offsets.reserve(additional);
        self.validity.reserve(additional);
    }

    fn finish(&mut self) -> ArrayRef {
        self.finish_into_varbin().into_array()
    }

    fn finish_into_canonical(&mut self, ctx: &mut ExecutionCtx) -> Canonical {
        self.finish()
            .execute::<Canonical>(ctx)
            .vortex_expect("varbin builder should canonicalize")
    }
}

/// Recovers a concrete [`VarBinBuilder`] from a `&mut dyn ArrayBuilder`.
///
/// Evaluates to `Some($body)` with `$typed` bound to the `&mut VarBinBuilder<O>` when the builder
/// is one, and `None` otherwise, so an encoding can fall through to its other output paths:
///
/// ```ignore
/// if let Some(result) = match_each_varbin_builder!(builder, |builder| {
///     append_my_encoding(array, builder, ctx)
/// }) {
///     return result;
/// }
/// ```
#[macro_export]
macro_rules! match_each_varbin_builder {
    ($builder:expr, | $typed:ident | $body:expr) => {
        $crate::__match_varbin_builder_widths!($builder, |$typed| $body, [u32, u64, i32, i64])
    };
}

/// Expands `$body` once per listed offset width, guarded by a downcast. See
/// [`match_each_varbin_builder!`].
#[doc(hidden)]
#[macro_export]
macro_rules! __match_varbin_builder_widths {
    ($builder:expr, | $typed:ident | $body:expr, [$($width:ty),+ $(,)?]) => {{
        let __varbin_builder: &mut dyn $crate::builders::ArrayBuilder = $builder;
        $crate::__match_varbin_builder_arms!(__varbin_builder, |$typed| $body, [$($width),+])
    }};
}

/// The `if`/`else if` chain behind [`__match_varbin_builder_widths!`], one arm per width.
#[doc(hidden)]
#[macro_export]
macro_rules! __match_varbin_builder_arms {
    ($builder:expr, | $typed:ident | $body:expr, []) => {
        None
    };
    ($builder:expr, | $typed:ident | $body:expr, [$head:ty $(, $tail:ty)*]) => {
        if $builder
            .as_any()
            .is::<$crate::builders::VarBinBuilder<$head>>()
        {
            let $typed = match $builder
                .as_any_mut()
                .downcast_mut::<$crate::builders::VarBinBuilder<$head>>()
            {
                Some(typed) => typed,
                None => unreachable!("builder type checked above"),
            };
            Some($body)
        } else {
            $crate::__match_varbin_builder_arms!($builder, |$typed| $body, [$($tail),*])
        }
    };
}

/// Running totals of `lengths`, wrapping so a corrupt lengths child is rejected rather than
/// panicking; [`VarBinBuilder::extend_offsets`] catches the resulting non-monotonic offsets.
#[inline]
fn prefix_sums<P: AsPrimitive<usize>>(lengths: &[P]) -> impl Iterator<Item = usize> {
    lengths.iter().scan(0usize, |end, length| {
        *end = end.wrapping_add(length.as_());
        Some(*end)
    })
}

#[cfg(test)]
mod tests {
    use std::mem::MaybeUninit;

    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::varbin::VarBinArraySlotsExt;
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
        let mut builder = VarBinBuilder::<i32>::with_capacity(DType::Utf8(Nullable), 0);
        builder.append(Some(b"hello"));
        builder.append(None);
        builder.append(Some(b"world"));
        let array = builder.finish_into_varbin();

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
        let mut ctx = array_session().create_execution_ctx();

        let actual = with_offsets(large_offsets, source.dtype().clone(), |builder| {
            source.append_to_builder(builder, &mut ctx)
        })?;

        assert_arrays_eq!(actual, source, &mut ctx);
        Ok(())
    }

    #[test]
    fn append_n_values_offset_overflow_returns_error() {
        let mut builder = VarBinBuilder::<i32>::new(DType::Utf8(Nullable));

        // The limit is checked before anything is reserved, so the huge byte total is free.
        let result = builder.append_n_values(b"hello", i32::MAX as usize / 5 + 1);

        assert!(result.is_err());
        assert_eq!(builder.offsets.len(), 1);
        assert!(builder.data.is_empty());
        assert_eq!(builder.validity.len(), 0);
    }

    #[test]
    fn append_values_rejects_a_short_offset_count() {
        let mut builder = VarBinBuilder::<i32>::new(DType::Utf8(Nullable));

        let result = builder.append_values(b"ab", [1usize, 2].into_iter(), &Mask::new_true(3));

        assert!(result.is_err());
        assert_eq!(builder.offsets.len(), 1);
        assert!(builder.data.is_empty());
    }

    #[test]
    fn append_values_rejects_non_monotonic_offsets() {
        let mut builder = VarBinBuilder::<i32>::new(DType::Utf8(Nullable));

        let result = builder.append_values(b"ab", [2usize, 1].into_iter(), &Mask::new_true(2));

        assert!(result.is_err());
        assert_eq!(builder.offsets.len(), 1);
    }

    #[test]
    fn append_decoded_writes_into_the_builder_storage() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let mut builder = VarBinBuilder::<i32>::new(DType::Utf8(Nullable));

        // SAFETY: the closure initializes exactly the 6 bytes it reports.
        unsafe {
            builder.append_decoded(
                6,
                4,
                &[3usize, 0, 3],
                &Mask::from_iter([true, false, true]),
                &mut |spare: &mut [MaybeUninit<u8>]| {
                    for (slot, byte) in spare.iter_mut().zip(b"foobar") {
                        slot.write(*byte);
                    }
                    Ok(6)
                },
            )?;
        }

        let expected =
            VarBinViewArray::from_iter([Some("foo"), None, Some("bar")], DType::Utf8(Nullable));
        assert_arrays_eq!(builder.finish_into_varbin(), expected, &mut ctx);
        Ok(())
    }

    /// The offset width has to be checked before `decode` runs: an overflowing `i32` builder
    /// should not pay for a full decompression first.
    #[test]
    fn append_decoded_rejects_an_offset_overflow_without_decoding() {
        let mut builder = VarBinBuilder::<i32>::new(DType::Utf8(Nullable));
        let num_bytes = i32::MAX as usize + 1;
        let mut decoded = false;

        // SAFETY: the closure is never called, and reports only what it initializes if it were.
        let result = unsafe {
            builder.append_decoded(num_bytes, 0, &[num_bytes], &Mask::new_true(1), &mut |_| {
                decoded = true;
                Ok(0)
            })
        };

        assert!(result.is_err());
        assert!(!decoded, "decode ran before the offset width was checked");
        assert_eq!(builder.offsets.len(), 1);
        assert_eq!(builder.validity.len(), 0);
    }

    /// `append_scalar_repeated` reports an offset overflow rather than panicking, so a constant
    /// column wider than the offset type errors like the bulk appends do.
    #[test]
    fn append_scalar_repeated_rejects_an_offset_overflow() {
        let mut builder = VarBinBuilder::<i32>::new(DType::Utf8(Nullable));

        let result = builder
            .append_scalar_repeated(&Scalar::utf8("hello", Nullable), i32::MAX as usize / 5 + 1);

        assert!(result.is_err());
        assert_eq!(builder.offsets.len(), 1);
        assert!(builder.data.is_empty());
        assert_eq!(builder.validity.len(), 0);
    }

    #[test]
    fn append_decoded_rejects_a_short_decode() {
        let mut builder = VarBinBuilder::<i32>::new(DType::Utf8(Nullable));

        // SAFETY: the closure initializes the 3 bytes it reports (none, and it reports 3 — but the
        // length mismatch is rejected before anything is published).
        let result = unsafe {
            builder.append_decoded(6, 0, &[3usize, 3], &Mask::new_true(2), &mut |spare| {
                spare[..3].fill(MaybeUninit::new(b'x'));
                Ok(3)
            })
        };

        assert!(result.is_err());
        assert_eq!(builder.offsets.len(), 1);
        assert_eq!(builder.validity.len(), 0);
    }

    /// Slices that overrun the declared byte count are rejected, and the builder is left exactly
    /// as it was — the overrun bytes are copied before the total can be checked, so the unwind has
    /// to put them back.
    #[rstest]
    #[case::all_valid(Mask::new_true(2))]
    #[case::some_valid(Mask::from_iter([true, false, true]))]
    fn append_valid_slices_rejects_a_byte_count_mismatch(#[case] validity: Mask) {
        let mut builder = VarBinBuilder::<i32>::new(DType::Utf8(Nullable));
        let values = [b"foo".as_slice(), b"quux".as_slice()];

        let mut next = 0;
        let result = builder.append_valid_slices(6, &validity, |_| {
            next += 1;
            values[next - 1]
        });

        assert!(result.is_err());
        assert_eq!(builder.offsets.len(), 1);
        assert!(builder.data.is_empty());
        assert_eq!(builder.validity.len(), 0);
    }

    #[test]
    #[should_panic(expected = "The offset count must be one more than the validity length")]
    fn finish_rejects_mismatched_validity() {
        let mut builder = VarBinBuilder::<i32>::new(DType::Utf8(Nullable));
        builder.validity.append_true();
        drop(builder.finish_into_varbin());
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
            let result = with_offsets(large_offsets, DType::Utf8(Nullable), |builder| {
                builder.reserve_exact(3);
                builder.append_zero();
                builder.append_scalar(&Scalar::utf8("hello", Nullable))?;
                builder.append_null();
                assert_eq!(builder.len(), 3);
                Ok(())
            })?;
            assert_eq!(result.validity()?.execute_mask(3, &mut ctx)?, validity);
        }
        Ok(())
    }

    #[rstest]
    #[case(false)]
    #[case(true)]
    fn test_append_varbinview_validity_to_builder(#[case] large_offsets: bool) -> VortexResult<()> {
        let long = "a value that does not fit inline";
        let all_null = VarBinViewArray::from_iter([None::<&str>, None], DType::Utf8(Nullable));
        let mixed =
            VarBinViewArray::from_iter([Some("hello"), None, Some(long)], DType::Utf8(Nullable));
        let expected = VarBinViewArray::from_iter(
            [None, None, Some("hello"), None, Some(long)],
            DType::Utf8(Nullable),
        );
        let mut ctx = array_session().create_execution_ctx();

        let actual = with_offsets(large_offsets, expected.dtype().clone(), |builder| {
            all_null
                .clone()
                .into_array()
                .append_to_builder(builder, &mut ctx)?;
            mixed
                .clone()
                .into_array()
                .append_to_builder(builder, &mut ctx)
        })?;

        assert_arrays_eq!(actual, expected, &mut ctx);
        Ok(())
    }

    /// `match_each_varbin_builder!` covers every offset width a `VarBinBuilder` can be built with,
    /// so both canonical appends must reach their specialization for all of them. A macro that
    /// only matched the signed pair would send an unsigned builder down a downcast that assumes
    /// `VarBinViewBuilder` and panic.
    #[rstest]
    #[case::u32(VarBinBuilder::<u32>::new(DType::Utf8(Nullable)))]
    #[case::u64(VarBinBuilder::<u64>::new(DType::Utf8(Nullable)))]
    #[case::i32(VarBinBuilder::<i32>::new(DType::Utf8(Nullable)))]
    #[case::i64(VarBinBuilder::<i64>::new(DType::Utf8(Nullable)))]
    fn append_to_every_offset_width(#[case] mut builder: impl ArrayBuilder) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let long = "a string that is far too long to be inlined in a view";
        let values = [Some("hello"), None, Some(long), Some("")];

        let view = VarBinViewArray::from_iter(values, DType::Utf8(Nullable)).into_array();
        let varbin = VarBinArray::from_iter(values, DType::Utf8(Nullable)).into_array();
        let constant = ConstantArray::new(Scalar::from("hello").into_nullable(), 2).into_array();

        for chunk in [&view, &varbin, &constant] {
            chunk.append_to_builder(&mut builder, &mut ctx)?;
        }

        let expected = ChunkedArray::try_new(vec![view, varbin, constant], DType::Utf8(Nullable))?;
        assert_arrays_eq!(builder.finish(), expected, &mut ctx);
        Ok(())
    }

    #[test]
    fn offsets_have_is_sorted_stat() -> VortexResult<()> {
        let mut builder = VarBinBuilder::<i32>::with_capacity(DType::Utf8(Nullable), 0);
        builder.append_value(b"aaa");
        builder.push_null();
        builder.append_value(b"bbb");
        let array = builder.finish_into_varbin();

        let is_sorted = array
            .offsets()
            .statistics()
            .with_typed_stats_set(|s| s.get_as::<bool>(Stat::IsSorted));
        assert_eq!(is_sorted, Precision::Exact(true));
        Ok(())
    }

    #[test]
    fn empty_builder_offsets_have_is_sorted_stat() -> VortexResult<()> {
        let mut builder = VarBinBuilder::<i32>::new(DType::Utf8(Nullable));
        let array = builder.finish_into_varbin();

        let is_sorted = array
            .offsets()
            .statistics()
            .with_typed_stats_set(|s| s.get_as::<bool>(Stat::IsSorted));
        assert_eq!(is_sorted, Precision::Exact(true));
        Ok(())
    }

    /// Runs `f` against an `i32` or `i64` builder and returns the finished array.
    fn with_offsets(
        large_offsets: bool,
        dtype: DType,
        f: impl FnOnce(&mut dyn ArrayBuilder) -> VortexResult<()>,
    ) -> VortexResult<VarBinArray> {
        if large_offsets {
            let mut builder = VarBinBuilder::<i64>::with_capacity(dtype, 8);
            f(&mut builder)?;
            Ok(builder.finish_into_varbin())
        } else {
            let mut builder = VarBinBuilder::<i32>::with_capacity(dtype, 8);
            f(&mut builder)?;
            Ok(builder.finish_into_varbin())
        }
    }
}
