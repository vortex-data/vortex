// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod bool;
mod decimal;
mod extension;
mod fixed_size_list;
mod list_view;
mod null;
mod primitive;
mod struct_;
mod union;
mod varbinview;

use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::mem::size_of;

use bool::bool_uncompressed_size_in_bytes;
use decimal::decimal_uncompressed_size_in_bytes;
use extension::extension_uncompressed_size_in_bytes;
use fixed_size_list::fixed_size_list_uncompressed_size_in_bytes;
use list_view::list_view_uncompressed_size_in_bytes;
use null::null_uncompressed_size_in_bytes;
use primitive::primitive_uncompressed_size_in_bytes;
use prost::Message;
use struct_::struct_uncompressed_size_in_bytes;
use union::union_uncompressed_size_in_bytes;
use varbinview::varbinview_uncompressed_size_in_bytes;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::Canonical;
use crate::Columnar;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::AggregateArgs;
use crate::aggregate_fn::AggregateFnId;
use crate::aggregate_fn::AggregateFnVTable;
use crate::aggregate_fn::DynAccumulator;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::ListView;
use crate::arrays::map::MapArraySlotsExt;
use crate::arrays::varbinview::BinaryView;
use crate::dtype::DType;
use crate::dtype::Nullability::NonNullable;
use crate::dtype::PType;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::proto::expr as pb;
use crate::scalar::Scalar;

/// Whether [`UncompressedSizeInBytes`] approximates the bytes an array uses or counts them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum UncompressedSizePrecision {
    /// An approximation: every byte of every buffer the canonical array holds, whether or not a
    /// row uses it. Cheap, but a view array sharing or sliced from a larger buffer reports the
    /// whole buffer.
    #[default]
    Inexact,
    /// The actual number of bytes the array's rows use, which costs a pass over view and list
    /// arrays.
    Exact,
}

/// Options for [`UncompressedSizeInBytes`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct UncompressedSizeOpts {
    pub precision: UncompressedSizePrecision,
}

impl UncompressedSizeOpts {
    /// Approximate the bytes the array uses by the buffers it holds. This is the default.
    pub const fn inexact() -> Self {
        Self {
            precision: UncompressedSizePrecision::Inexact,
        }
    }

    /// Count the actual bytes the array's rows use.
    pub const fn exact() -> Self {
        Self {
            precision: UncompressedSizePrecision::Exact,
        }
    }

    pub const fn is_exact(&self) -> bool {
        matches!(self.precision, UncompressedSizePrecision::Exact)
    }

    /// Serialize these options to protobuf-encoded metadata bytes.
    pub fn serialize(&self) -> Vec<u8> {
        pb::UncompressedSizeInBytesOpts {
            exact: self.is_exact(),
        }
        .encode_to_vec()
    }

    /// Deserialize these options from protobuf-encoded metadata bytes.
    pub fn deserialize(metadata: &[u8]) -> VortexResult<Self> {
        let opts = pb::UncompressedSizeInBytesOpts::decode(metadata)?;
        Ok(if opts.exact {
            Self::exact()
        } else {
            Self::inexact()
        })
    }
}

impl Display for UncompressedSizeOpts {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // Only the non-default configuration is displayed, so the default renders as
        // `vortex.uncompressed_size_in_bytes()`.
        if self.is_exact() {
            write!(f, "precision=exact")?;
        }
        Ok(())
    }
}

/// Return the uncompressed size of an array in bytes.
///
/// See [`UncompressedSizeInBytes`] for details and [`UncompressedSizeOpts`] for the choice of
/// precision. The result is cached as a statistic on `array` and on any children measured.
pub fn uncompressed_size_in_bytes(
    array: &ArrayRef,
    opts: UncompressedSizeOpts,
    ctx: &mut ExecutionCtx,
) -> VortexResult<usize> {
    let size = uncompressed_size_in_bytes_u64(array, opts, ctx)?;

    usize::try_from(size)
        .map_err(|e| vortex_err!("Failed to convert uncompressed size to usize: {e}"))
}

fn uncompressed_size_in_bytes_u64(
    array: &ArrayRef,
    opts: UncompressedSizeOpts,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    // The stat slot holds only the exact size. The inexact size is a different measure, which can
    // be larger or smaller than the exact one, so it neither reads nor writes the slot: reusing an
    // exact value would make it depend on what was computed before, and caching it would put a
    // value in the slot that bounds nothing.
    if opts.is_exact()
        && let Precision::Exact(size) = array.statistics().get(Stat::UncompressedSizeInBytes)
    {
        return u64::try_from(&size)
            .map_err(|e| e.with_context("Failed to convert uncompressed size stat to u64"));
    }

    // The accumulator caches an exact size on `array` through the stat slot.
    let mut acc = Accumulator::try_new(UncompressedSizeInBytes, opts, array.dtype().clone())?;
    acc.accumulate(array, ctx)?;
    let result = acc.finish()?;

    Ok(result
        .as_primitive()
        .typed_value::<u64>()
        .vortex_expect("uncompressed_size_in_bytes result should not be null"))
}

/// The byte size of all buffers in children in their canonical representation.
///
/// Applies to all types and returns a non-null `u64`. Encoding kernels can return this aggregate
/// directly from metadata to avoid decoding arrays whose uncompressed size is known.
///
/// By default the size is [inexact](UncompressedSizePrecision::Inexact): an approximation that
/// counts every buffer the canonical array holds in full. The
/// [exact](UncompressedSizePrecision::Exact) size is the actual number of bytes the array's rows
/// use, at the cost of a pass over view and list arrays.
///
/// This is generally useful for various execution engines to pick better join orderings.
#[derive(Clone, Debug)]
pub struct UncompressedSizeInBytes;

impl UncompressedSizeInBytes {
    /// Whether arrays of `dtype` can be measured. Variant arrays cannot.
    pub fn supports_dtype(dtype: &DType) -> bool {
        supports_uncompressed_size_in_bytes(dtype)
    }
}

impl AggregateFnVTable for UncompressedSizeInBytes {
    type Options = UncompressedSizeOpts;
    type Partial = u64;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.uncompressed_size_in_bytes");
        *ID
    }

    fn serialize(&self, options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(options.serialize()))
    }

    fn deserialize(
        &self,
        metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        UncompressedSizeOpts::deserialize(metadata)
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        supports_uncompressed_size_in_bytes(input_dtype)
            .then_some(DType::Primitive(PType::U64, NonNullable))
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
    ) -> VortexResult<Self::Partial> {
        Ok(0)
    }

    fn partial_from_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        scalar: &Scalar,
    ) -> VortexResult<Self::Partial> {
        Ok(scalar
            .as_primitive()
            .typed_value::<u64>()
            .vortex_expect("uncompressed_size_in_bytes partial should not be null"))
    }

    fn merge_partials(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        first: Self::Partial,
        second: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        first
            .checked_add(second)
            .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))
    }

    fn to_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        Ok(Scalar::primitive(*partial, NonNullable))
    }

    #[inline]
    fn is_saturated(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _partial: &Self::Partial,
    ) -> bool {
        false
    }

    fn accumulate(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let opts = *args.options;
        let size = match batch {
            Columnar::Canonical(canonical) => {
                canonical_uncompressed_size_in_bytes(canonical, opts, ctx)?
            }
            Columnar::Constant(constant) => {
                constant_uncompressed_size_in_bytes(constant.as_view(), opts, ctx)?
            }
        };
        *partial = partial
            .checked_add(size)
            .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))?;
        Ok(())
    }

    fn finalize(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partials: ArrayRef,
    ) -> VortexResult<ArrayRef> {
        Ok(partials)
    }

    fn finalize_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        self.to_scalar(args, partial)
    }
}

/// Measure a canonical array.
pub(crate) fn canonical_uncompressed_size_in_bytes(
    canonical: &Canonical,
    opts: UncompressedSizeOpts,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    match canonical {
        Canonical::Null(array) => Ok(null_uncompressed_size_in_bytes(array)),
        Canonical::Bool(array) => bool_uncompressed_size_in_bytes(array, ctx),
        Canonical::Primitive(array) => primitive_uncompressed_size_in_bytes(array, ctx),
        Canonical::Decimal(array) => decimal_uncompressed_size_in_bytes(array, ctx),
        Canonical::VarBinView(array) => varbinview_uncompressed_size_in_bytes(array, opts, ctx),
        Canonical::List(array) => list_view_uncompressed_size_in_bytes(array, opts, ctx),
        Canonical::Map(array) => list_view_uncompressed_size_in_bytes(
            &array.entries().as_::<ListView>().into_owned(),
            opts,
            ctx,
        ),
        Canonical::FixedSizeList(array) => {
            fixed_size_list_uncompressed_size_in_bytes(array, opts, ctx)
        }
        Canonical::Struct(array) => struct_uncompressed_size_in_bytes(array, opts, ctx),
        Canonical::Union(array) => union_uncompressed_size_in_bytes(array, opts, ctx),
        Canonical::Extension(array) => extension_uncompressed_size_in_bytes(array, opts, ctx),
        Canonical::Variant(_) => {
            vortex_bail!("UncompressedSizeInBytes is not supported for Variant arrays")
        }
    }
}

pub(crate) fn constant_uncompressed_size_in_bytes(
    array: ArrayView<'_, Constant>,
    opts: UncompressedSizeOpts,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    let value_size = match array.dtype() {
        DType::Null => return Ok(0),
        DType::Bool(_) => packed_bit_buffer_size_in_bytes(array.len())?,
        DType::Primitive(ptype, _) => {
            checked_len_mul(array.len(), ptype.byte_width(), "primitive")?
        }
        DType::Decimal(decimal_type, _) => checked_len_mul(
            array.len(),
            decimal_type.canonical_values_type().byte_width(),
            "decimal",
        )?,
        DType::Utf8(_) => constant_varbinview_value_size(
            array.len(),
            array.scalar().as_utf8().value().map(|value| value.len()),
            opts,
        )?,
        DType::Binary(_) => constant_varbinview_value_size(
            array.len(),
            array.scalar().as_binary().value().map(|value| value.len()),
            opts,
        )?,
        DType::List(..)
        | DType::Map(..)
        | DType::FixedSizeList(..)
        | DType::Struct(..)
        | DType::Union(..)
        | DType::Extension(_) => {
            let canonical = array.array().clone().execute::<Canonical>(ctx)?;
            return canonical_uncompressed_size_in_bytes(&canonical, opts, ctx);
        }
        DType::Variant(_) => {
            vortex_bail!("UncompressedSizeInBytes is not supported for Variant arrays")
        }
    };

    value_size
        .checked_add(constant_validity_size(array, ctx)?)
        .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))
}

fn constant_varbinview_value_size(
    len: usize,
    scalar_len: Option<usize>,
    opts: UncompressedSizeOpts,
) -> VortexResult<u64> {
    let views_size = checked_len_mul(len, size_of::<BinaryView>(), "binary view")?;
    // Only a value too long to inline adds a data buffer, matching `constant_canonicalize`. Every
    // row references it, so the exact size counts it once per row, like a canonical array's
    // views, while the buffer canonicalization creates holds it once.
    let data_size = match scalar_len {
        Some(scalar_len) if scalar_len > BinaryView::MAX_INLINED_SIZE => {
            let rows = if opts.is_exact() { len } else { 1 };
            checked_len_mul(rows, scalar_len, "binary data")?
        }
        _ => 0,
    };

    views_size
        .checked_add(data_size)
        .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))
}

fn constant_validity_size(
    array: ArrayView<'_, Constant>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    let validity = array.validity()?.execute_mask(array.len(), ctx)?;
    validity_uncompressed_size_in_bytes(validity)
}

fn checked_len_mul(len: usize, width: usize, name: &str) -> VortexResult<u64> {
    let len = u64::try_from(len)
        .map_err(|e| vortex_err!("Failed to convert {name} length to u64: {e}"))?;
    let width = u64::try_from(width)
        .map_err(|e| vortex_err!("Failed to convert {name} byte width to u64: {e}"))?;

    len.checked_mul(width)
        .ok_or_else(|| vortex_err!("uncompressed size in bytes overflowed u64"))
}

fn supports_uncompressed_size_in_bytes(dtype: &DType) -> bool {
    match dtype {
        DType::Null
        | DType::Bool(_)
        | DType::Primitive(..)
        | DType::Decimal(..)
        | DType::Utf8(_)
        | DType::Binary(_) => true,
        DType::List(element_dtype, _) | DType::FixedSizeList(element_dtype, ..) => {
            supports_uncompressed_size_in_bytes(element_dtype)
        }
        DType::Map(map_dtype, _) => {
            supports_uncompressed_size_in_bytes(&map_dtype.key_dtype())
                && supports_uncompressed_size_in_bytes(&map_dtype.value_dtype())
        }
        DType::Struct(fields, _) => fields
            .fields()
            .all(|field| supports_uncompressed_size_in_bytes(&field)),
        DType::Union(variants, _) => variants
            .variants()
            .all(|variant| supports_uncompressed_size_in_bytes(&variant)),
        DType::Variant(_) => false,
        DType::Extension(ext_dtype) => {
            supports_uncompressed_size_in_bytes(ext_dtype.storage_dtype())
        }
    }
}

pub(crate) fn validity_uncompressed_size_in_bytes(validity: Mask) -> VortexResult<u64> {
    match validity {
        Mask::AllTrue(_) => Ok(0),
        Mask::AllFalse(len) => Ok(ConstantArray::new(false, len).into_array().nbytes()),
        Mask::Values(values) => packed_bit_buffer_size_in_bytes(values.len()),
    }
}

pub(crate) fn packed_bit_buffer_size_in_bytes(len: usize) -> VortexResult<u64> {
    u64::try_from(len.div_ceil(8))
        .map_err(|e| vortex_err!("Failed to convert bit buffer length to u64: {e}"))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::RecursiveCanonical;
    use crate::VortexSessionExecute;
    use crate::aggregate_fn::Accumulator;
    use crate::aggregate_fn::AggregateDTypes;
    use crate::aggregate_fn::AggregateFnVTable;
    use crate::aggregate_fn::AggregateFnVTableExt;
    use crate::aggregate_fn::DynAccumulator;
    use crate::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
    use crate::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeOpts;
    use crate::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::ExtensionArray;
    use crate::arrays::FixedSizeListArray;
    use crate::arrays::ListViewArray;
    use crate::arrays::NullArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::StructArray;
    use crate::arrays::UnionArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::VariantArray;
    use crate::arrays::listview::ListViewRebuildMode;
    use crate::arrays::varbinview::BinaryView;
    use crate::builders::builder_with_capacity_in;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::FieldNames;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::UnionVariants;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::expr::stats::StatsProvider;
    use crate::extension::datetime::Date;
    use crate::extension::datetime::TimeUnit;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    /// The size the array occupies once rebuilt through the canonical builders, which is the
    /// layout [`UncompressedSizeInBytes`] is defined against: the builders normalize physical
    /// widths that the input array is free to choose differently, picking the canonical decimal
    /// width of a precision and `u64` list-view offsets and sizes.
    ///
    /// Builders no longer canonicalize their children, so the finished array is only canonical at
    /// the top level - recursively canonicalize it before measuring.
    fn materialized_uncompressed_size_in_bytes(array: &ArrayRef) -> u64 {
        let mut ctx = SESSION.create_execution_ctx();
        let mut builder = builder_with_capacity_in(
            array.dtype(),
            array.len(),
            vortex_buffer::BufferAllocatorRef::static_ref(),
        );
        array
            .append_to_builder(builder.as_mut(), &mut ctx)
            .vortex_expect("appended");
        builder
            .finish()
            .execute::<RecursiveCanonical>(&mut ctx)
            .vortex_expect("recursively canonicalized")
            .0
            .into_array()
            .nbytes()
    }

    fn aggregate(array: &ArrayRef) -> VortexResult<u64> {
        aggregate_with(array, UncompressedSizeOpts::default())
    }

    fn aggregate_with(array: &ArrayRef, opts: UncompressedSizeOpts) -> VortexResult<u64> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut acc = Accumulator::try_new(UncompressedSizeInBytes, opts, array.dtype().clone())?;
        acc.accumulate(array, &mut ctx)?;
        acc.finish()?
            .as_primitive()
            .typed_value::<u64>()
            .ok_or_else(|| vortex_err!("uncompressed size result should not be null"))
    }

    /// A view array whose rows point into one shared data buffer, like Arrow's Parquet reader
    /// builds, sliced to `rows`.
    fn views_into_shared_page(rows: std::ops::Range<usize>) -> VortexResult<ArrayRef> {
        const VALUE_LEN: u32 = 27;
        let page = ByteBuffer::from(vec![b'x'; 1000 * VALUE_LEN as usize]);
        let views = (0..1000u32)
            .map(|row| BinaryView::make_view(&[b'x'; VALUE_LEN as usize], 0, row * VALUE_LEN))
            .collect::<Buffer<BinaryView>>();
        // SAFETY: every view points at `VALUE_LEN` bytes inside the page.
        let page = unsafe {
            VarBinViewArray::new_unchecked(
                views,
                Arc::from([page]),
                DType::Utf8(Nullability::NonNullable),
                Validity::NonNullable,
            )
        };
        page.into_array().slice(rows)
    }

    #[test]
    fn primitive_matches_materialized_size() -> VortexResult<()> {
        let array = PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        assert_eq!(aggregate(&array)?, 16);
        Ok(())
    }

    #[test]
    fn nullable_primitive_matches_materialized_size() -> VortexResult<()> {
        let array = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        assert_eq!(aggregate(&array)?, 13);
        Ok(())
    }

    #[test]
    fn all_invalid_primitive_matches_materialized_size() -> VortexResult<()> {
        let array = PrimitiveArray::new(buffer![0i32, 0, 0], Validity::AllInvalid).into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        assert_eq!(aggregate(&array)?, 14);
        Ok(())
    }

    #[test]
    fn bool_matches_materialized_size() -> VortexResult<()> {
        let array = BoolArray::from_iter([true, false, true, true, false]).into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        assert_eq!(aggregate(&array)?, 1);
        Ok(())
    }

    #[test]
    fn nullable_bool_matches_materialized_size() -> VortexResult<()> {
        let array = BoolArray::from_iter([Some(true), None, Some(false), Some(true)]).into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        assert_eq!(aggregate(&array)?, 2);
        Ok(())
    }

    #[test]
    fn all_invalid_bool_matches_materialized_size() -> VortexResult<()> {
        let array = BoolArray::from_iter([None::<bool>, None, None]).into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        assert_eq!(aggregate(&array)?, 3);
        Ok(())
    }

    #[test]
    fn null_matches_materialized_size() -> VortexResult<()> {
        let array = NullArray::new(5).into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        assert_eq!(aggregate(&array)?, 0);
        Ok(())
    }

    #[test]
    fn decimal_matches_materialized_size() -> VortexResult<()> {
        let array = DecimalArray::new(
            buffer![12345i64, -123i64, 0i64],
            DecimalDType::new(5, 2),
            Validity::NonNullable,
        )
        .into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        assert_eq!(aggregate(&array)?, 12);
        Ok(())
    }

    #[test]
    fn varbinview_matches_materialized_size() -> VortexResult<()> {
        let array = VarBinViewArray::from_iter_nullable_str([
            Some("short"),
            None,
            Some("this string is longer than twelve bytes"),
        ])
        .into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        assert_eq!(aggregate(&array)?, 88);
        Ok(())
    }

    #[test]
    fn list_matches_materialized_size() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let elements =
            PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();
        let offsets = buffer![2u32, 0].into_array();
        let sizes = buffer![2u32, 1].into_array();
        let array = ListViewArray::new(elements, offsets, sizes, Validity::NonNullable, &mut ctx)
            .into_array();

        // The lists are out of order and leave element 1 unreferenced, so compare against the
        // exact layout rather than a builder round-trip.
        let mut ctx = SESSION.create_execution_ctx();
        let exact = array
            .clone()
            .execute::<ListViewArray>(&mut ctx)?
            .rebuild(ListViewRebuildMode::MakeExact, &mut ctx)?
            .into_array();

        assert_eq!(materialized_uncompressed_size_in_bytes(&exact), 44);
        assert_eq!(
            aggregate_with(&array, UncompressedSizeOpts::exact())?,
            materialized_uncompressed_size_in_bytes(&exact)
        );
        // The inexact size counts the unreferenced element too.
        assert_eq!(
            aggregate_with(&array, UncompressedSizeOpts::inexact())?,
            materialized_uncompressed_size_in_bytes(&exact) + 4
        );
        Ok(())
    }

    #[test]
    fn varbinview_exact_counts_referenced_bytes_only() -> VortexResult<()> {
        let sliced = views_into_shared_page(100..200)?;

        // 100 views of 16 bytes plus 100 values of 27 bytes.
        assert_eq!(
            aggregate_with(&sliced, UncompressedSizeOpts::exact())?,
            100 * (16 + 27)
        );
        // The inexact size still charges the whole 27,000-byte page.
        assert_eq!(
            aggregate_with(&sliced, UncompressedSizeOpts::inexact())?,
            100 * 16 + 1000 * 27
        );
        Ok(())
    }

    #[test]
    fn varbinview_exact_skips_null_rows_and_inlined_views() -> VortexResult<()> {
        let array = VarBinViewArray::from_iter_nullable_str([
            Some("inlined"),
            None,
            Some("this string is longer than twelve bytes"),
        ])
        .into_array();
        let nulled = VarBinViewArray::from_iter_nullable_str([Some("inlined"), None, None::<&str>])
            .into_array();

        // Three views, one validity byte, and only the one out-of-line value.
        assert_eq!(
            aggregate_with(&array, UncompressedSizeOpts::exact())?,
            3 * 16 + 1 + 39
        );
        // Nulling the long value drops its bytes from the exact size.
        assert_eq!(
            aggregate_with(&nulled, UncompressedSizeOpts::exact())?,
            3 * 16 + 1
        );
        Ok(())
    }

    #[test]
    fn exact_matches_inexact_for_fixed_width_types() -> VortexResult<()> {
        let arrays = [
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array(),
            BoolArray::from_iter([Some(true), None, Some(false)]).into_array(),
            DecimalArray::new(
                buffer![12345i64, -123i64, 0i64],
                DecimalDType::new(5, 2),
                Validity::NonNullable,
            )
            .into_array(),
            NullArray::new(5).into_array(),
        ];
        for (array, expected) in arrays.into_iter().zip([13, 2, 12, 0]) {
            assert_eq!(
                aggregate_with(&array, UncompressedSizeOpts::exact())?,
                expected
            );
            assert_eq!(
                aggregate_with(&array, UncompressedSizeOpts::exact())?,
                aggregate_with(&array, UncompressedSizeOpts::inexact())?
            );
        }
        Ok(())
    }

    #[test]
    fn nested_exact_measures_children_exactly() -> VortexResult<()> {
        let strings = views_into_shared_page(0..10)?;
        let array = StructArray::try_new(
            FieldNames::from(["strings"]),
            vec![strings.clone()],
            10,
            Validity::NonNullable,
        )?
        .into_array();

        // Measure inexactly first: an exact result is cached on the shared child, and an inexact
        // query would then reuse it.
        let inexact = aggregate_with(&array, UncompressedSizeOpts::inexact())?;
        assert_eq!(
            inexact,
            aggregate_with(&strings, UncompressedSizeOpts::inexact())?
        );
        let exact = aggregate_with(&array, UncompressedSizeOpts::exact())?;
        assert_eq!(
            exact,
            aggregate_with(&strings, UncompressedSizeOpts::exact())?
        );
        assert!(exact < inexact);
        assert_eq!(exact, 430);
        assert_eq!(inexact, 27160);
        Ok(())
    }

    #[test]
    fn exact_result_is_cached_as_exact_stat() -> VortexResult<()> {
        let sliced = views_into_shared_page(0..10)?;
        let mut ctx = SESSION.create_execution_ctx();

        let size = uncompressed_size_in_bytes(&sliced, UncompressedSizeOpts::exact(), &mut ctx)?;

        assert_eq!(
            sliced.statistics().get(Stat::UncompressedSizeInBytes),
            Precision::exact(u64::try_from(size)?)
        );
        assert_eq!(size, 430);
        Ok(())
    }

    #[test]
    fn inexact_result_is_not_cached_and_ignores_exact_stat() -> VortexResult<()> {
        let sliced = views_into_shared_page(0..10)?;
        let mut ctx = SESSION.create_execution_ctx();

        let inexact =
            uncompressed_size_in_bytes(&sliced, UncompressedSizeOpts::inexact(), &mut ctx)?;
        assert!(
            sliced
                .statistics()
                .get(Stat::UncompressedSizeInBytes)
                .is_absent()
        );

        // An exact query computes and caches the exact size.
        let exact = uncompressed_size_in_bytes(&sliced, UncompressedSizeOpts::exact(), &mut ctx)?;
        assert_eq!(
            sliced.statistics().get(Stat::UncompressedSizeInBytes),
            Precision::exact(u64::try_from(exact)?)
        );

        // An inexact query still measures the held buffers rather than reusing the exact size.
        assert_eq!(
            uncompressed_size_in_bytes(&sliced, UncompressedSizeOpts::inexact(), &mut ctx)?,
            inexact
        );
        assert_eq!(exact, 430);
        assert_eq!(inexact, 27160);
        Ok(())
    }

    #[test]
    fn inexact_can_be_smaller_than_exact() -> VortexResult<()> {
        // Every row references the same out-of-line value, which the data buffer holds once.
        const VALUE_LEN: u32 = 20;
        let page = ByteBuffer::from(vec![b'x'; VALUE_LEN as usize]);
        let views = (0..10)
            .map(|_| BinaryView::make_view(&[b'x'; VALUE_LEN as usize], 0, 0))
            .collect::<Buffer<BinaryView>>();
        // SAFETY: every view points at the `VALUE_LEN` bytes of the page.
        let one_value = unsafe {
            VarBinViewArray::new_unchecked(
                views,
                Arc::from([page]),
                DType::Utf8(Nullability::NonNullable),
                Validity::NonNullable,
            )
        }
        .into_array();

        let exact = aggregate_with(&one_value, UncompressedSizeOpts::exact())?;
        let inexact = aggregate_with(&one_value, UncompressedSizeOpts::inexact())?;
        assert_eq!(exact, 10 * (16 + 20));
        assert_eq!(inexact, 10 * 16 + 20);
        Ok(())
    }

    #[test]
    fn only_exact_aggregate_has_stat_slot() {
        assert_eq!(
            Stat::from_aggregate_fn(&UncompressedSizeInBytes.bind(UncompressedSizeOpts::exact())),
            Some(Stat::UncompressedSizeInBytes)
        );
        assert_eq!(
            Stat::from_aggregate_fn(&UncompressedSizeInBytes.bind(UncompressedSizeOpts::inexact())),
            None
        );
    }

    #[test]
    fn options_round_trip_and_display() -> VortexResult<()> {
        for opts in [
            UncompressedSizeOpts::exact(),
            UncompressedSizeOpts::inexact(),
        ] {
            assert_eq!(UncompressedSizeOpts::deserialize(&opts.serialize())?, opts);
        }
        assert_eq!(UncompressedSizeOpts::exact().to_string(), "precision=exact");
        assert_eq!(UncompressedSizeOpts::inexact().to_string(), "");
        Ok(())
    }

    #[test]
    fn fixed_size_list_matches_materialized_size() -> VortexResult<()> {
        let elements =
            PrimitiveArray::from_option_iter([Some(1i32), None, Some(3), Some(4)]).into_array();
        let array = FixedSizeListArray::new(elements, 2, Validity::NonNullable, 2).into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        assert_eq!(aggregate(&array)?, 17);
        Ok(())
    }

    #[test]
    fn struct_matches_materialized_size() -> VortexResult<()> {
        let ints = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
        let strings = VarBinViewArray::from_iter_nullable_str([Some("alpha"), None, Some("omega")])
            .into_array();
        let array = StructArray::try_new(
            FieldNames::from(["ints", "strings"]),
            vec![ints, strings],
            3,
            Validity::NonNullable,
        )?
        .into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        assert_eq!(aggregate(&array)?, 62);
        Ok(())
    }

    #[test]
    fn union_sums_type_ids_and_sparse_children() -> VortexResult<()> {
        let type_ids = PrimitiveArray::from_iter([5_u8, 9, 5]).into_array();
        let numbers = PrimitiveArray::from_iter([10_i32, 0, 30]).into_array();
        let flags = BoolArray::from_iter([false, true, false]).into_array();
        let expected = aggregate(&type_ids)? + aggregate(&numbers)? + aggregate(&flags)?;
        let variants = UnionVariants::try_new(
            ["number", "flag"].into(),
            vec![
                DType::Primitive(PType::I32, Nullability::NonNullable),
                DType::Bool(Nullability::NonNullable),
            ],
            vec![5, 9],
        )?;
        let array = UnionArray::try_new(type_ids, variants, vec![numbers, flags])?.into_array();

        assert_eq!(aggregate(&array)?, expected);
        assert_eq!(expected, 16);
        Ok(())
    }

    #[test]
    fn extension_matches_materialized_size() -> VortexResult<()> {
        let storage = PrimitiveArray::from_option_iter([Some(1i32), None, Some(3)]).into_array();
        let ext_dtype = Date::new(TimeUnit::Days, Nullability::Nullable).erased();
        let array = ExtensionArray::new(ext_dtype, storage).into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        assert_eq!(aggregate(&array)?, 13);
        Ok(())
    }

    #[test]
    fn variant_stat_is_unsupported() -> VortexResult<()> {
        let child = ConstantArray::new(Scalar::variant(Scalar::from(42i32)), 3).into_array();
        let array = VariantArray::try_new(child, None)?.into_array();
        let mut ctx = SESSION.create_execution_ctx();

        assert_eq!(
            array
                .statistics()
                .compute_uncompressed_size_in_bytes(&mut ctx),
            None
        );
        Ok(())
    }

    #[test]
    fn constant_matches_materialized_size() -> VortexResult<()> {
        let array = ConstantArray::new(42i32, 10).into_array();

        assert_eq!(
            aggregate(&array)?,
            materialized_uncompressed_size_in_bytes(&array)
        );
        assert_eq!(aggregate(&array)?, 40);
        Ok(())
    }

    #[test]
    fn chunked_sums_chunk_sizes() -> VortexResult<()> {
        let chunk1 = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::NonNullable).into_array();
        let chunk2 = PrimitiveArray::new(buffer![4i32, 5], Validity::NonNullable).into_array();
        let expected = materialized_uncompressed_size_in_bytes(&chunk1)
            + materialized_uncompressed_size_in_bytes(&chunk2);
        let chunked = ChunkedArray::try_new(
            vec![chunk1, chunk2],
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )?
        .into_array();

        assert_eq!(aggregate(&chunked)?, expected);
        assert_eq!(expected, 20);
        Ok(())
    }

    #[test]
    fn size_with_cached_exact_stat() -> VortexResult<()> {
        let array = ConstantArray::new(42i32, 10).into_array();
        array.statistics().set(
            Stat::UncompressedSizeInBytes,
            Precision::Exact(ScalarValue::from(40u64)),
        );

        assert_eq!(aggregate_with(&array, UncompressedSizeOpts::exact())?, 40);
        Ok(())
    }

    #[test]
    fn helper_caches_result() -> VortexResult<()> {
        let array = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::NonNullable).into_array();
        let mut ctx = SESSION.create_execution_ctx();

        let size = uncompressed_size_in_bytes(&array, UncompressedSizeOpts::exact(), &mut ctx)?;

        assert_eq!(
            array.statistics().get(Stat::UncompressedSizeInBytes),
            Precision::exact(u64::try_from(size)?)
        );
        assert_eq!(size, 12);
        Ok(())
    }

    #[test]
    fn state_merge() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let opts = UncompressedSizeOpts::default();
        let dtypes = AggregateDTypes::try_new(&UncompressedSizeInBytes, &opts, dtype)?;

        let state = UncompressedSizeInBytes.merge_partials(dtypes.args(&opts), 5, 3)?;

        let result = UncompressedSizeInBytes.to_scalar(dtypes.args(&opts), &state)?;
        assert_eq!(result.as_primitive().typed_value::<u64>(), Some(8));
        Ok(())
    }
}
