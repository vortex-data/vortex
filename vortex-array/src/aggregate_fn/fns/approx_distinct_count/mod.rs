// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An approximate count of the distinct non-null values in an array.

use std::hash::BuildHasher;

use num_traits::ToPrimitive;
use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;
use vortex_utils::aliases::DefaultHashBuilder;
use vortex_utils::aliases::hash_map::HashTable;
use vortex_utils::aliases::hash_set::HashSet;

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
use crate::aggregate_fn::EmptyOptions;
use crate::arrays::ConstantArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::varbinview::BinaryView;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::Nullability::NonNullable;
use crate::dtype::PType;
use crate::dtype::ToBytes;
use crate::match_each_decimal_value_type;
use crate::match_each_native_ptype;
use crate::scalar::Scalar;

/// The largest decimal precision whose values all fit in an `i128`.
const MAX_DECIMAL_PRECISION: u8 = 38;

/// Approximately count the distinct non-null values in `array`.
pub fn approx_distinct_count(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<u64> {
    let mut acc = Accumulator::try_new(ApproxDistinctCount, EmptyOptions, array.dtype().clone())?;
    acc.accumulate(array, ctx)?;
    let result = acc.finish()?;
    Ok(result
        .as_primitive()
        .typed_value::<u64>()
        .vortex_expect("approx_distinct_count result should not be null"))
}

/// Approximately count the distinct non-null values of an input.
///
/// The partial state is either exact, holding the set of distinct values seen so far, or inexact,
/// holding only a count. Merging two exact states unions their sets; merging when either side is
/// inexact adds the two counts, so an inexact count is an upper bound on the distinct values.
///
/// Accumulating and merging in memory, as the writer does across a whole file, stays exact: the
/// set holds every distinct value, so its memory grows with the column's cardinality. Only the
/// serialized partial, which the file footer stores, drops the set and keeps the count, so a
/// partial read back from a scalar is inexact. Values hash by their physical
/// representation, so floats compare by their bits (`-0.0` and `0.0` are distinct, as are NaNs
/// with different payloads).
///
/// Applies to boolean, primitive, decimal (up to precision 38), UTF-8 and binary inputs, and to
/// extension types over those, and returns a non-null `u64`.
#[derive(Clone, Debug)]
pub struct ApproxDistinctCount;

impl ApproxDistinctCount {
    /// Whether the aggregate can be applied to `dtype`.
    pub fn supports_dtype(dtype: &DType) -> bool {
        key_kind(dtype).is_some()
    }
}

/// The partial state of [`ApproxDistinctCount`].
#[derive(Clone, Debug)]
pub enum ApproxDistinctCountPartial {
    /// The set of every distinct value seen.
    Exact(DistinctValues),
    /// An upper bound on the number of distinct values seen.
    Inexact(u64),
}

impl ApproxDistinctCountPartial {
    /// The number of distinct values, exact or not.
    pub fn count(&self) -> u64 {
        match self {
            Self::Exact(values) => values.len() as u64,
            Self::Inexact(count) => *count,
        }
    }

    /// Whether the count is exact.
    pub fn is_exact(&self) -> bool {
        matches!(self, Self::Exact(_))
    }
}

/// The distinct values of an exact [`ApproxDistinctCountPartial`].
#[derive(Clone, Debug)]
pub enum DistinctValues {
    /// Fixed-width values (booleans, primitives and decimals) zero-extended to 128 bits.
    Fixed(HashSet<u128>),
    /// Variable-length values (UTF-8 and binary).
    Bytes(ByteSet),
}

impl DistinctValues {
    fn len(&self) -> usize {
        match self {
            Self::Fixed(set) => set.len(),
            Self::Bytes(set) => set.len(),
        }
    }

    fn union(self, other: Self) -> VortexResult<Self> {
        Ok(match (self, other) {
            (Self::Fixed(a), Self::Fixed(b)) => Self::Fixed(union(a, b)),
            (Self::Bytes(a), Self::Bytes(b)) => Self::Bytes(a.union(b)),
            _ => vortex_bail!("Cannot merge distinct values of different kinds"),
        })
    }
}

fn union<T: Eq + std::hash::Hash>(a: HashSet<T>, b: HashSet<T>) -> HashSet<T> {
    let (mut larger, smaller) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    larger.extend(smaller);
    larger
}

#[derive(Clone, Copy)]
enum KeyKind {
    Fixed,
    Bytes,
}

fn key_kind(dtype: &DType) -> Option<KeyKind> {
    match dtype {
        DType::Bool(_) | DType::Primitive(..) => Some(KeyKind::Fixed),
        DType::Decimal(decimal, _) => {
            (decimal.precision() <= MAX_DECIMAL_PRECISION).then_some(KeyKind::Fixed)
        }
        DType::Utf8(_) | DType::Binary(_) => Some(KeyKind::Bytes),
        DType::Extension(ext) => key_kind(ext.storage_dtype()),
        _ => None,
    }
}

impl AggregateFnVTable for ApproxDistinctCount {
    type Options = EmptyOptions;
    type Partial = ApproxDistinctCountPartial;

    fn id(&self) -> AggregateFnId {
        static ID: CachedId = CachedId::new("vortex.approx_distinct_count");
        *ID
    }

    fn serialize(&self, _options: &Self::Options) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        _metadata: &[u8],
        _session: &VortexSession,
    ) -> VortexResult<Self::Options> {
        Ok(EmptyOptions)
    }

    fn return_dtype(&self, _options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        Self::supports_dtype(input_dtype).then_some(DType::Primitive(PType::U64, NonNullable))
    }

    fn partial_dtype(&self, options: &Self::Options, input_dtype: &DType) -> Option<DType> {
        self.return_dtype(options, input_dtype)
    }

    fn empty_partial(&self, args: AggregateArgs<'_, Self::Options>) -> VortexResult<Self::Partial> {
        let Some(kind) = key_kind(args.dtype) else {
            vortex_bail!("ApproxDistinctCount does not support dtype {}", args.dtype);
        };
        Ok(ApproxDistinctCountPartial::Exact(match kind {
            KeyKind::Fixed => DistinctValues::Fixed(HashSet::default()),
            KeyKind::Bytes => DistinctValues::Bytes(ByteSet::default()),
        }))
    }

    fn partial_from_scalar(
        &self,
        args: AggregateArgs<'_, Self::Options>,
        scalar: &Scalar,
    ) -> VortexResult<Self::Partial> {
        let count = scalar
            .as_primitive()
            .typed_value::<u64>()
            .vortex_expect("approx_distinct_count partial should not be null");
        // A count of zero saw no values, which is exact.
        if count == 0 {
            return self.empty_partial(args);
        }
        Ok(ApproxDistinctCountPartial::Inexact(count))
    }

    fn merge_partials(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        first: Self::Partial,
        second: Self::Partial,
    ) -> VortexResult<Self::Partial> {
        use ApproxDistinctCountPartial::Exact;
        use ApproxDistinctCountPartial::Inexact;

        Ok(match (first, second) {
            (Exact(a), Exact(b)) => Exact(a.union(b)?),
            (a, b) => Inexact(a.count() + b.count()),
        })
    }

    /// Only the count is serialized: the distinct values are dropped.
    fn to_scalar(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &Self::Partial,
    ) -> VortexResult<Scalar> {
        Ok(Scalar::primitive(partial.count(), NonNullable))
    }

    fn is_saturated(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        _partial: &Self::Partial,
    ) -> bool {
        false
    }

    fn accumulate(
        &self,
        _args: AggregateArgs<'_, Self::Options>,
        partial: &mut Self::Partial,
        batch: &Columnar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        accumulate_columnar(partial, batch, ctx)
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

/// Receives the hashing keys of the non-null values of a batch.
///
/// Fixed-width values (booleans, primitives and decimals) arrive zero-extended to 128 bits, and
/// variable-length values (UTF-8 and binary) as their bytes.
trait DistinctSink {
    fn add_fixed(&mut self, keys: impl Iterator<Item = u128>) -> VortexResult<()>;

    fn add_bytes<'a>(&mut self, values: impl Iterator<Item = &'a [u8]>) -> VortexResult<()>;
}

/// Accumulate the non-null values of a columnar batch into `sink`.
fn accumulate_columnar<S: DistinctSink>(
    sink: &mut S,
    batch: &Columnar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match batch {
        Columnar::Constant(c) => {
            if c.is_empty() || c.scalar().is_null() {
                return Ok(());
            }
            let single = ConstantArray::new(c.scalar().clone(), 1)
                .into_array()
                .execute::<Canonical>(ctx)?;
            accumulate_canonical(sink, &single, ctx)
        }
        Columnar::Canonical(c) => accumulate_canonical(sink, c, ctx),
    }
}

fn accumulate_canonical<S: DistinctSink>(
    partial: &mut S,
    canonical: &Canonical,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    match canonical {
        Canonical::Null(_) => Ok(()),
        Canonical::Bool(b) => {
            let Some(valid) = valid_bits(b.as_ref(), ctx)? else {
                return Ok(());
            };
            let bits = b.to_bit_buffer();
            partial.add_fixed(filter_valid(bits.iter().map(u128::from), valid.as_ref()))
        }
        Canonical::Primitive(p) => {
            let Some(valid) = valid_bits(p.as_ref(), ctx)? else {
                return Ok(());
            };
            match_each_native_ptype!(p.ptype(), |T| {
                partial.add_fixed(filter_valid(
                    p.as_slice::<T>().iter().map(fixed_key::<T>),
                    valid.as_ref(),
                ))
            })
        }
        Canonical::Decimal(d) => {
            let Some(valid) = valid_bits(d.as_ref(), ctx)? else {
                return Ok(());
            };
            match_each_decimal_value_type!(d.values_type(), |D| {
                let values = d.buffer::<D>();
                partial.add_fixed(filter_valid(
                    values.iter().map(|v| {
                        // Precision is at most `MAX_DECIMAL_PRECISION`, so values fit.
                        v.to_i128().vortex_expect("decimal value fits in i128") as u128
                    }),
                    valid.as_ref(),
                ))
            })
        }
        Canonical::VarBinView(v) => {
            let Some(valid) = valid_bits(v.as_ref(), ctx)? else {
                return Ok(());
            };
            partial.add_bytes(filter_valid(
                v.views().iter().map(|view| view_bytes(v, view)),
                valid.as_ref(),
            ))
        }
        Canonical::Extension(e) => {
            let storage = e.storage_array().clone().execute::<Canonical>(ctx)?;
            accumulate_canonical(partial, &storage, ctx)
        }
        _ => vortex_bail!(
            "ApproxDistinctCount does not support dtype {}",
            canonical.dtype()
        ),
    }
}

/// The validity of `array`: `None` if every value is null, `Some(None)` if every value is valid.
fn valid_bits(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<Option<BitBuffer>>> {
    Ok(match array.validity()?.execute_mask(array.len(), ctx)? {
        Mask::AllTrue(_) => Some(None),
        Mask::AllFalse(_) => None,
        Mask::Values(values) => Some(Some(values.bit_buffer().clone())),
    })
}

fn filter_valid<'a, T>(
    values: impl Iterator<Item = T> + 'a,
    valid: Option<&'a BitBuffer>,
) -> impl Iterator<Item = T> + 'a {
    values
        .enumerate()
        .filter(move |(idx, _)| valid.is_none_or(|valid| valid.value(*idx)))
        .map(|(_, value)| value)
}

fn view_bytes<'a>(array: &'a VarBinViewArray, view: &'a BinaryView) -> &'a [u8] {
    if view.is_inlined() {
        view.as_inlined().value()
    } else {
        let view_ref = view.as_view();
        &array.buffer(view_ref.buffer_index as usize).as_slice()[view_ref.as_range()]
    }
}

fn fixed_key<T: NativePType>(value: &T) -> u128 {
    let bytes = ToBytes::to_le_bytes(value);
    let mut key = [0u8; 16];
    key[..bytes.len()].copy_from_slice(bytes);
    u128::from_le_bytes(key)
}

impl DistinctSink for ApproxDistinctCountPartial {
    fn add_fixed(&mut self, keys: impl Iterator<Item = u128>) -> VortexResult<()> {
        match self {
            ApproxDistinctCountPartial::Exact(DistinctValues::Fixed(set)) => set.extend(keys),
            ApproxDistinctCountPartial::Inexact(count) => {
                *count += keys.collect::<HashSet<_>>().len() as u64;
            }
            ApproxDistinctCountPartial::Exact(DistinctValues::Bytes(_)) => {
                vortex_bail!("Expected fixed-width distinct values")
            }
        }
        Ok(())
    }

    fn add_bytes<'a>(&mut self, values: impl Iterator<Item = &'a [u8]>) -> VortexResult<()> {
        match self {
            ApproxDistinctCountPartial::Exact(DistinctValues::Bytes(set)) => {
                values.for_each(|value| set.insert(value));
            }
            ApproxDistinctCountPartial::Inexact(count) => {
                let mut set = ByteSet::default();
                values.for_each(|value| set.insert(value));
                *count += set.len() as u64;
            }
            ApproxDistinctCountPartial::Exact(DistinctValues::Fixed(_)) => {
                vortex_bail!("Expected variable-length distinct values")
            }
        }
        Ok(())
    }
}

/// A set of byte strings stored back to back in one buffer, so inserting a value allocates
/// nothing per value.
#[derive(Clone, Default)]
pub struct ByteSet {
    /// The concatenated bytes of every value.
    bytes: Vec<u8>,
    /// The end offset of each value in `bytes`.
    ends: Vec<usize>,
    /// Indices into `ends`, keyed by the hash of the value's bytes.
    table: HashTable<usize>,
    hasher: DefaultHashBuilder,
}

impl std::fmt::Debug for ByteSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ByteSet").field("len", &self.len()).finish()
    }
}

impl ByteSet {
    fn len(&self) -> usize {
        self.ends.len()
    }

    fn value(&self, idx: usize) -> &[u8] {
        let start = idx.checked_sub(1).map_or(0, |prev| self.ends[prev]);
        &self.bytes[start..self.ends[idx]]
    }

    fn insert(&mut self, value: &[u8]) {
        let hash = self.hasher.hash_one(value);
        let Self {
            bytes,
            ends,
            table,
            hasher,
        } = self;
        let value_at = |idx: usize| {
            let start = idx.checked_sub(1).map_or(0, |prev| ends[prev]);
            &bytes[start..ends[idx]]
        };
        if table.find(hash, |&idx| value_at(idx) == value).is_some() {
            return;
        }
        let idx = ends.len();
        table.insert_unique(hash, idx, |&idx| {
            let start = idx.checked_sub(1).map_or(0, |prev| ends[prev]);
            hasher.hash_one(&bytes[start..ends[idx]])
        });
        bytes.extend_from_slice(value);
        ends.push(bytes.len());
    }

    fn union(self, other: Self) -> Self {
        let (mut larger, smaller) = if self.len() >= other.len() {
            (self, other)
        } else {
            (other, self)
        };
        for idx in 0..smaller.len() {
            larger.insert(smaller.value(idx));
        }
        larger
    }
}

mod hll;

pub use hll::ApproxDistinctCountHll;
pub use hll::ApproxDistinctCountHllPartial;

#[cfg(test)]
mod tests;
