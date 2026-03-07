// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fastlanes::FastLanes;
use vortex_array::ArrayCommon;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_array::stats::ArrayStats;
use vortex_error::VortexExpect as _;

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
/// use vortex_fastlanes::DeltaVTable;
/// let array = DeltaVTable::try_from_vec(vec![1_u32, 2, 3, 5, 10, 11]).unwrap();
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
    pub(super) common: ArrayCommon,
    pub(super) bases: ArrayRef,
    pub(super) deltas: ArrayRef,
}

/// Extension trait for [`DeltaArray`] methods.
pub trait DeltaArrayExt {
    fn bases(&self) -> &ArrayRef;

    fn deltas(&self) -> &ArrayRef;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool;

    fn dtype(&self) -> &DType;

    /// The logical offset into the first chunk of [`Self::deltas`].
    fn offset(&self) -> usize;
}

impl DeltaArrayExt for DeltaArray {
    #[inline]
    fn bases(&self) -> &ArrayRef {
        &self.bases
    }

    #[inline]
    fn deltas(&self) -> &ArrayRef {
        &self.deltas
    }

    #[inline]
    fn len(&self) -> usize {
        self.common.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.common.len() == 0
    }

    #[inline]
    fn dtype(&self) -> &DType {
        self.common.dtype()
    }

    #[inline]
    /// The logical offset into the first chunk of [`Self::deltas`].
    fn offset(&self) -> usize {
        self.offset
    }
}

impl DeltaArray {
    pub(crate) unsafe fn new_unchecked(
        bases: ArrayRef,
        deltas: ArrayRef,
        offset: usize,
        logical_len: usize,
    ) -> Self {
        let dtype = bases.dtype().with_nullability(deltas.dtype().nullability());
        Self {
            offset,
            common: ArrayCommon::new(logical_len, dtype),
            bases,
            deltas,
        }
    }

    #[inline]
    pub(crate) fn lanes(&self) -> usize {
        let ptype =
            PType::try_from(self.dtype()).vortex_expect("DeltaArray DType must be primitive");
        lane_count(ptype)
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
        self.common.stats()
    }
}

pub(crate) fn lane_count(ptype: PType) -> usize {
    match_each_unsigned_integer_ptype!(ptype, |T| { T::LANES })
}
