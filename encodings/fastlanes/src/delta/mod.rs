use std::fmt::Debug;
use std::sync::{Arc, RwLock};

pub use compress::*;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::stats::StatsSet;
use vortex_array::validity::Validity;
use vortex_array::variants::PrimitiveArrayTrait;
use vortex_array::vtable::{StatisticsVTable, VTableRef};
use vortex_array::{
    encoding_ids, Array, ArrayCanonicalImpl, ArrayImpl, ArrayRef, ArrayStatisticsImpl,
    ArrayValidityImpl, ArrayVariantsImpl, Canonical, Encoding, EncodingId, RkyvMetadata,
};
use vortex_buffer::Buffer;
use vortex_dtype::{match_each_unsigned_integer_ptype, DType, NativePType, PType};
use vortex_error::{vortex_bail, VortexExpect as _, VortexResult};
use vortex_mask::Mask;

use crate::delta::serde::DeltaMetadata;

mod compress;
mod compute;
mod serde;

#[derive(Clone, Debug)]
pub struct DeltaArray {
    offset: usize,
    len: usize,
    dtype: DType,
    bases: ArrayRef,
    deltas: ArrayRef,
    validity: Validity,
    stats_set: Arc<RwLock<StatsSet>>,
}

pub struct DeltaEncoding;
impl Encoding for DeltaEncoding {
    const ID: EncodingId = EncodingId::new("fastlanes.delta", encoding_ids::FL_DELTA);
    type Array = DeltaArray;
    type Metadata = RkyvMetadata<DeltaMetadata>;
}

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
/// contains 1,024 vlaues, then there are as many bases as there are _lanes_ of this type in a
/// 1024-bit register. For example, for 64-bit values, there are 16 bases because there are 16
/// _lanes_. Each lane is a [delta-encoding](https://en.wikipedia.org/wiki/Delta_encoding) `1024 /
/// bit_width` long vector of values. The deltas are stored in the
/// [FastLanes](https://www.vldb.org/pvldb/vol16/p2132-afroozeh.pdf) order which splits the 1,024
/// values into one contiguous sub-sequence per-lane, thus permitting delta encoding.
///
/// If the chunk physically has fewer than 1,024 values, then it is stored as a traditional,
/// non-SIMD-amenable, delta-encoded vector.
impl DeltaArray {
    // TODO(ngates): remove constructing from vec
    pub fn try_from_vec<T: NativePType>(vec: Vec<T>) -> VortexResult<Self> {
        Self::try_from_primitive_array(&PrimitiveArray::new(
            Buffer::copy_from(vec),
            Validity::NonNullable,
        ))
    }

    pub fn try_from_primitive_array(array: &PrimitiveArray) -> VortexResult<Self> {
        let (bases, deltas) = delta_compress(array)?;

        Self::try_from_delta_compress_parts(
            bases.into_array(),
            deltas.into_array(),
            Validity::NonNullable,
        )
    }

    pub fn try_from_delta_compress_parts(
        bases: ArrayRef,
        deltas: ArrayRef,
        validity: Validity,
    ) -> VortexResult<Self> {
        let logical_len = deltas.len();
        Self::try_new(bases, deltas, validity, 0, logical_len)
    }

    pub fn try_new(
        bases: ArrayRef,
        deltas: ArrayRef,
        validity: Validity,
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
        if bases.dtype() != deltas.dtype() {
            vortex_bail!(
                "DeltaArray: bases and deltas must have the same dtype, got {:?} and {:?}",
                bases.dtype(),
                deltas.dtype()
            );
        }
        let dtype = bases.dtype().clone();
        if !dtype.is_int() {
            vortex_bail!("DeltaArray: dtype must be an integer, got {}", dtype);
        }

        if let Some(vlen) = validity.maybe_len() {
            if vlen != logical_len {
                vortex_bail!(
                    "DeltaArray: validity length ({}) must match logical_len ({})",
                    vlen,
                    logical_len
                );
            }
        }

        let delta = Self {
            offset,
            len: logical_len,
            dtype,
            bases,
            deltas,
            validity,
            stats_set: Default::default(),
        };

        if delta.bases().len() != delta.bases_len() {
            vortex_bail!(
                "DeltaArray: bases.len() ({}) != expected_bases_len ({}), based on len ({}) and lane count ({})",
                delta.bases().len(),
                delta.bases_len(),
                logical_len,
                delta.lanes()
            );
        }

        if (delta.deltas_len() % 1024 == 0) != (delta.bases_len() % delta.lanes() == 0) {
            vortex_bail!(
                "deltas length ({}) is a multiple of 1024 iff bases length ({}) is a multiple of LANES ({})",
                delta.deltas_len(),
                delta.bases_len(),
                delta.lanes(),
            );
        }

        Ok(delta)
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
    fn lanes(&self) -> usize {
        let ptype = PType::try_from(self.dtype())
            .vortex_expect("Failed to convert DeltaArray DType to PType");
        match_each_unsigned_integer_ptype!(ptype, |$T| {
            <$T as fastlanes::FastLanes>::LANES
        })
    }

    #[inline]
    /// The logical offset into the first chunk of [`Self::deltas`].
    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn validity(&self) -> &Validity {
        &self.validity
    }

    fn bases_len(&self) -> usize {
        self.bases.len()
    }

    fn deltas_len(&self) -> usize {
        self.deltas.len()
    }
}

impl ArrayImpl for DeltaArray {
    type Encoding = DeltaEncoding;

    fn _len(&self) -> usize {
        self.len
    }

    fn _dtype(&self) -> &DType {
        &self.dtype
    }

    fn _vtable(&self) -> VTableRef {
        VTableRef::new_ref(&DeltaEncoding)
    }
}

impl ArrayCanonicalImpl for DeltaArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        delta_decompress(self).map(Canonical::Primitive)
    }
}

impl ArrayStatisticsImpl for DeltaArray {
    fn _stats_set(&self) -> &RwLock<StatsSet> {
        &self.stats_set
    }
}

impl ArrayValidityImpl for DeltaArray {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        self.validity.is_valid(index)
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        self.validity.all_valid()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        self.validity.all_invalid()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        self.validity.to_logical(self.len)
    }
}

impl ArrayVariantsImpl for DeltaArray {
    fn _as_primitive_typed(&self) -> Option<&dyn PrimitiveArrayTrait> {
        Some(self)
    }
}

impl PrimitiveArrayTrait for DeltaArray {}

impl StatisticsVTable<&DeltaArray> for DeltaEncoding {}
