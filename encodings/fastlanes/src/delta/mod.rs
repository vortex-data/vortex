use std::fmt::{Debug, Display};

pub use compress::*;
use serde::{Deserialize, Serialize};
use vortex_array::array::PrimitiveArray;
use vortex_array::encoding::ids;
use vortex_array::stats::{StatisticsVTable, StatsSet};
use vortex_array::validity::{LogicalValidity, Validity, ValidityMetadata, ValidityVTable};
use vortex_array::variants::{ArrayVariants, PrimitiveArrayTrait};
use vortex_array::visitor::{ArrayVisitor, VisitorVTable};
use vortex_array::{
    impl_encoding, ArrayDType, ArrayData, ArrayLen, ArrayTrait, Canonical, IntoArrayData,
    IntoCanonical,
};
use vortex_dtype::{match_each_unsigned_integer_ptype, NativePType};
use vortex_error::{vortex_bail, vortex_panic, VortexExpect as _, VortexResult};

mod compress;
mod compute;

impl_encoding!("fastlanes.delta", ids::FL_DELTA, Delta);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaMetadata {
    validity: ValidityMetadata,
    deltas_len: u64,
    offset: u16, // must be <1024
}

impl Display for DeltaMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self, f)
    }
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
    pub fn try_from_vec<T: NativePType>(vec: Vec<T>) -> VortexResult<Self> {
        Self::try_from_primitive_array(&PrimitiveArray::from(vec))
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
        bases: ArrayData,
        deltas: ArrayData,
        validity: Validity,
    ) -> VortexResult<Self> {
        let logical_len = deltas.len();
        Self::try_new(bases, deltas, validity, 0, logical_len)
    }

    pub fn try_new(
        bases: ArrayData,
        deltas: ArrayData,
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

        let metadata = DeltaMetadata {
            validity: validity.to_metadata(logical_len)?,
            deltas_len: deltas.len() as u64,
            offset: offset as u16,
        };

        let mut children = vec![bases, deltas];
        if let Some(varray) = validity.into_array() {
            children.push(varray)
        }

        let delta = Self::try_from_parts(
            dtype,
            logical_len,
            metadata,
            children.into(),
            StatsSet::default(),
        )?;

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
    pub fn bases(&self) -> ArrayData {
        self.as_ref()
            .child(0, self.dtype(), self.bases_len())
            .vortex_expect("DeltaArray is missing bases child array")
    }

    #[inline]
    pub fn deltas(&self) -> ArrayData {
        self.as_ref()
            .child(1, self.dtype(), self.deltas_len())
            .vortex_expect("DeltaArray is missing deltas child array")
    }

    #[inline]
    fn lanes(&self) -> usize {
        let ptype = self.dtype().try_into().unwrap_or_else(|err| {
            vortex_panic!(
                err,
                "Failed to convert DeltaArray DType {} to PType",
                self.dtype()
            )
        });
        match_each_unsigned_integer_ptype!(ptype, |$T| {
            <$T as fastlanes::FastLanes>::LANES
        })
    }

    #[inline]
    /// The logical offset into the first chunk of [`Self::deltas`].
    pub fn offset(&self) -> usize {
        self.metadata().offset as usize
    }

    pub fn validity(&self) -> Validity {
        self.metadata().validity.to_validity(|| {
            self.as_ref()
                .child(2, &Validity::DTYPE, self.len())
                .vortex_expect("DeltaArray: validity child")
        })
    }

    fn bases_len(&self) -> usize {
        let num_chunks = self.deltas().len() / 1024;
        let remainder_base_size = if self.deltas().len() % 1024 > 0 { 1 } else { 0 };
        num_chunks * self.lanes() + remainder_base_size
    }

    fn deltas_len(&self) -> usize {
        self.metadata().deltas_len as usize
    }
}

impl ArrayTrait for DeltaArray {}

impl ArrayVariants for DeltaArray {
    fn as_primitive_array(&self) -> Option<&dyn PrimitiveArrayTrait> {
        Some(self)
    }
}

impl PrimitiveArrayTrait for DeltaArray {}

impl IntoCanonical for DeltaArray {
    fn into_canonical(self) -> VortexResult<Canonical> {
        delta_decompress(self).map(Canonical::Primitive)
    }
}

impl ValidityVTable<DeltaArray> for DeltaEncoding {
    fn is_valid(&self, array: &DeltaArray, index: usize) -> bool {
        array.validity().is_valid(index)
    }

    fn logical_validity(&self, array: &DeltaArray) -> LogicalValidity {
        array.validity().to_logical(array.len())
    }
}

impl VisitorVTable<DeltaArray> for DeltaEncoding {
    fn accept(&self, array: &DeltaArray, visitor: &mut dyn ArrayVisitor) -> VortexResult<()> {
        visitor.visit_child("bases", &array.bases())?;
        visitor.visit_child("deltas", &array.deltas())
    }
}

impl StatisticsVTable<DeltaArray> for DeltaEncoding {}
