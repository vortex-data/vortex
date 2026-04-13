// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;

use fastlanes::FastLanes;
use vortex_array::ArrayRef;
use vortex_array::TypedArrayRef;
use vortex_array::dtype::PType;
use vortex_array::match_each_unsigned_integer_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

pub mod delta_compress;
pub mod delta_decompress;

/// The base values for each block of deltas.
pub(super) const BASES_SLOT: usize = 0;
/// The delta-encoded values relative to the base values.
pub(super) const DELTAS_SLOT: usize = 1;
pub(super) const NUM_SLOTS: usize = 2;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["bases", "deltas"];

/// A FastLanes-style delta-encoded array of primitive values.
///
/// A DeltaArray comprises a sequence of _chunks_ each representing exactly 1,024
/// delta-encoded values. If the input array length is not a multiple of 1,024, the last chunk
/// is padded with zeros to fill a complete 1,024-element chunk.
///
/// # Examples
///
/// ```
/// use vortex_array::arrays::PrimitiveArray;
/// use vortex_array::VortexSessionExecute;
/// use vortex_array::session::ArraySession;
/// use vortex_session::VortexSession;
/// use vortex_fastlanes::Delta;
///
/// let session = VortexSession::empty().with::<ArraySession>();
/// let primitive = PrimitiveArray::from_iter([1_u32, 2, 3, 5, 10, 11]);
/// let array = Delta::try_from_primitive_array(&primitive, &mut session.create_execution_ctx()).unwrap();
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
/// Each chunk is stored as a vector of bases and a vector of deltas. There are as many bases as
/// there are _lanes_ of this type in a 1024-bit register. For example, for 64-bit values, there
/// are 16 bases because there are 16 _lanes_. Each lane is a
/// [delta-encoding](https://en.wikipedia.org/wiki/Delta_encoding) `1024 / bit_width` long vector
/// of values. The deltas are stored in the
/// [FastLanes](https://www.vldb.org/pvldb/vol16/p2132-afroozeh.pdf) order which splits the 1,024
/// values into one contiguous sub-sequence per-lane, thus permitting delta encoding.
///
/// Note the validity is stored in the deltas array.
#[derive(Clone, Debug)]
pub struct DeltaData {
    pub(super) offset: usize,
}

impl Display for DeltaData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "offset: {}", self.offset)
    }
}

pub trait DeltaArrayExt: TypedArrayRef<crate::Delta> {
    fn bases(&self) -> &ArrayRef {
        self.as_ref().slots()[BASES_SLOT]
            .as_ref()
            .vortex_expect("DeltaArray bases slot")
    }

    fn deltas(&self) -> &ArrayRef {
        self.as_ref().slots()[DELTAS_SLOT]
            .as_ref()
            .vortex_expect("DeltaArray deltas slot")
    }

    fn offset(&self) -> usize {
        self.offset
    }
}

impl<T: TypedArrayRef<crate::Delta>> DeltaArrayExt for T {}

impl DeltaData {
    pub fn try_new(offset: usize) -> VortexResult<Self> {
        vortex_ensure!(offset < 1024, "offset must be less than 1024: {offset}");
        Ok(Self { offset })
    }
}

pub(crate) fn lane_count(ptype: PType) -> usize {
    match_each_unsigned_integer_ptype!(ptype, |T| { T::LANES })
}
