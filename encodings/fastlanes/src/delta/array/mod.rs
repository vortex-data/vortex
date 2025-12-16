// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::FastLanes;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::stats::ArrayStats;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_dtype::NativePType;
use vortex_dtype::PType;
use vortex_dtype::match_each_unsigned_integer_ptype;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

pub mod delta_compress;
pub mod delta_decompress;

/// A FastLanes-style delta-encoded array of primitive values.
///
/// A [`DeltaArray`] comprises a sequence of _chunks_ each representing 1,024 delta-encoded values,
/// except the last chunk which may represent from one to 1,024 values.
///
/// # Examples
///
/// ```
/// use vortex_fastlanes::DeltaArray;
/// let array = DeltaArray::try_from_vec(vec![1_u32, 2, 3, 5, 10, 11]).unwrap();
/// ```
///
/// # Details
///
/// To facilitate slicing, this array accepts an `offset` and `logical_len`. The offset must be
/// strictly less than 1,024 and the sum of `offset` and `logical_len` must not exceed the length of
/// the `deltas` array. These values permit logical slicing without modifying any chunk containing a
/// kept value. In particular, we may defer decompresison until the array is canonicalized or
/// indexed. The `offset` is a physical offset into the first chunk, which necessarily contains
/// 1,024 values. The `logical_len` is the number of logical values following the `offset`, which
/// may be less than the number of physically stored values.
///
/// Each chunk is stored as a vector of bases and a vector of deltas. If the chunk physically
/// contains 1,024 values, then there are as many bases as there are _lanes_ of this type in a
/// 1024-bit register. For example, for 64-bit values, there are 16 bases because there are 16
/// _lanes_. Each lane is a [delta-encoding](https://en.wikipedia.org/wiki/Delta_encoding) `1024 /
/// bit_width` long vector of values. The deltas are stored in the
/// [FastLanes](https://www.vldb.org/pvldb/vol16/p2132-afroozeh.pdf) order which splits the 1,024
/// values into one contiguous sub-sequence per-lane, thus permitting delta encoding.
///
/// If the chunk physically has fewer than 1,024 values, then it is stored as a traditional,
/// non-SIMD-amenable, delta-encoded vector.
///
/// Note the validity is stored in the deltas array.
#[derive(Clone, Debug)]
pub struct DeltaArray {
    pub(super) offset: usize,
    pub(super) len: usize,
    pub(super) dtype: DType,
    pub(super) bases: ArrayRef,
    pub(super) deltas: ArrayRef,
    pub(super) stats_set: ArrayStats,
}

impl DeltaArray {
    // TODO(ngates): remove constructing from vec
    pub fn try_from_vec<T: NativePType>(vec: Vec<T>) -> VortexResult<Self> {
        Self::try_from_primitive_array(&PrimitiveArray::new(
            Buffer::copy_from(vec),
            Validity::NonNullable,
        ))
    }

    pub fn try_from_primitive_array(array: &PrimitiveArray) -> VortexResult<Self> {
        let (bases, deltas) = delta_compress::delta_compress(array)?;

        Self::try_from_delta_compress_parts(bases.into_array(), deltas.into_array())
    }

    /// Create a [`DeltaArray`] from the given `bases` and `deltas` arrays.
    /// Note the `deltas` might be nullable
    pub fn try_from_delta_compress_parts(bases: ArrayRef, deltas: ArrayRef) -> VortexResult<Self> {
        let logical_len = deltas.len();
        Self::try_new(bases, deltas, 0, logical_len)
    }

    pub fn try_new(
        bases: ArrayRef,
        deltas: ArrayRef,
        offset: usize,
        logical_len: usize,
    ) -> VortexResult<Self> {
        if offset >= 1024 {
            vortex_bail!("offset must be less than 1024: {}", offset);
        }
        if offset + logical_len > deltas.len() {
            vortex_bail!(
                "offset + logical_len, {} + {}, must be less than or equal to the size of deltas: {}",
                offset,
                logical_len,
                deltas.len()
            )
        }
        if !bases.dtype().eq_ignore_nullability(deltas.dtype()) {
            vortex_bail!(
                "DeltaArray: bases and deltas must have the same dtype, got {:?} and {:?}",
                bases.dtype(),
                deltas.dtype()
            );
        }
        let DType::Primitive(ptype, _) = bases.dtype().clone() else {
            vortex_bail!(
                "DeltaArray: dtype must be an integer, got {}",
                bases.dtype()
            );
        };

        if !ptype.is_int() {
            vortex_bail!("DeltaArray: ptype must be an integer, got {}", ptype);
        }

        let lanes = lane_count(ptype);

        if deltas.len().is_multiple_of(1024) != bases.len().is_multiple_of(lanes) {
            vortex_bail!(
                "deltas length ({}) is a multiple of 1024 iff bases length ({}) is a multiple of LANES ({})",
                deltas.len(),
                bases.len(),
                lanes,
            );
        }

        // SAFETY: validation done above
        Ok(unsafe { Self::new_unchecked(bases, deltas, offset, logical_len) })
    }

    pub(crate) unsafe fn new_unchecked(
        bases: ArrayRef,
        deltas: ArrayRef,
        offset: usize,
        logical_len: usize,
    ) -> Self {
        Self {
            offset,
            len: logical_len,
            dtype: bases.dtype().with_nullability(deltas.dtype().nullability()),
            bases,
            deltas,
            stats_set: Default::default(),
        }
    }

    #[inline]
    pub fn bases(&self) -> &ArrayRef {
        &self.bases
    }

    #[inline]
    pub fn deltas(&self) -> &ArrayRef {
        &self.deltas
    }

    #[inline]
    pub(crate) fn lanes(&self) -> usize {
        let ptype =
            PType::try_from(self.dtype()).vortex_expect("DeltaArray DType must be primitive");
        lane_count(ptype)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    #[inline]
    /// The logical offset into the first chunk of [`Self::deltas`].
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline]
    pub(crate) fn bases_len(&self) -> usize {
        self.bases.len()
    }

    #[inline]
    pub(crate) fn deltas_len(&self) -> usize {
        self.deltas.len()
    }

    #[inline]
    pub(crate) fn stats_set(&self) -> &ArrayStats {
        &self.stats_set
    }
}

pub(crate) fn lane_count(ptype: PType) -> usize {
    match_each_unsigned_integer_ptype!(ptype, |T| { T::LANES })
}
