// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Flattened, slice-addressable views of canonical map arrays.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ListView;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::MapArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::map::MapArrayExt;
use vortex_array::arrays::map::MapArraySlotsExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::MapDType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_integer_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

/// Random access to the bytes of a [`VarBinViewArray`] without per-row handle clones.
pub(crate) struct Strings<'a> {
    pub views: &'a [BinaryView],
    pub buffers: Vec<&'a [u8]>,
}

impl<'a> Strings<'a> {
    pub fn new(array: &'a VarBinViewArray) -> Self {
        Self {
            views: array.views(),
            buffers: array
                .data_buffers()
                .iter()
                .map(|b| b.as_host().as_slice())
                .collect(),
        }
    }

    #[inline]
    pub fn get(&self, index: usize) -> &'a [u8] {
        self.views[index].bytes(&self.buffers)
    }
}

/// A canonical map decomposed into row offsets, keys and values.
pub(crate) struct FlatMap {
    pub map_dtype: MapDType,
    pub len: usize,
    pub offsets: Vec<usize>,
    pub sizes: Vec<usize>,
    pub validity: Validity,
    pub row_valid: Mask,
    pub keys: VarBinViewArray,
    pub values: ArrayRef,
}

pub(crate) fn to_usize_vec(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Vec<usize>> {
    let array = array.clone().execute::<PrimitiveArray>(ctx)?;
    Ok(match_each_integer_ptype!(array.ptype(), |P| {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        array.as_slice::<P>().iter().map(|&v| v as usize).collect()
    }))
}

impl FlatMap {
    pub fn new(map: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let map = map.clone().execute::<MapArray>(ctx)?;
        let map_dtype = map.map_dtype().clone();
        let len = map.len();
        let entries = map.entries().as_::<ListView>();
        let offsets = to_usize_vec(entries.offsets(), ctx)?;
        let sizes = to_usize_vec(entries.sizes(), ctx)?;
        let validity = map.map_validity();
        let row_valid = validity.execute_mask(len, ctx)?;
        let elements = entries.elements().clone().execute::<StructArray>(ctx)?;
        let keys = elements
            .unmasked_field(0)
            .clone()
            .execute::<VarBinViewArray>(ctx)?;
        let values = elements
            .unmasked_field(1)
            .clone()
            .execute::<Canonical>(ctx)?
            .into_array();
        Ok(Self {
            map_dtype,
            len,
            offsets,
            sizes,
            validity,
            row_valid,
            keys,
            values,
        })
    }

    /// Whether `row` has the same validity and the same entry range as the previous row.
    #[inline]
    pub fn repeats_prev(&self, row: usize) -> bool {
        row > 0
            && self.row_valid.value(row) == self.row_valid.value(row - 1)
            && (!self.row_valid.value(row)
                || (self.offsets[row] == self.offsets[row - 1]
                    && self.sizes[row] == self.sizes[row - 1]))
    }

    /// The entry range of a row, empty for null rows.
    #[inline]
    pub fn range(&self, row: usize) -> std::ops::Range<usize> {
        if self.row_valid.value(row) {
            self.offsets[row]..self.offsets[row] + self.sizes[row]
        } else {
            0..0
        }
    }
}

/// The indices of the set positions of a mask, as a non-nullable `u64` array.
pub(crate) fn mask_indices(mask: &Mask) -> ArrayRef {
    let indices: Buffer<u64> = match mask.indices() {
        AllOr::All => (0..mask.len() as u64).collect(),
        AllOr::None => Buffer::empty(),
        AllOr::Some(indices) => indices.iter().map(|&i| i as u64).collect(),
    };
    PrimitiveArray::new(indices, Validity::NonNullable).into_array()
}

/// Builds a list-view whose rows may share or skip entries.
pub(crate) fn build_listview(
    elements: ArrayRef,
    offsets: Vec<u64>,
    sizes: Vec<u64>,
    validity: Validity,
) -> VortexResult<ListViewArray> {
    let offsets = PrimitiveArray::new(Buffer::from(offsets), Validity::NonNullable).into_array();
    let sizes = PrimitiveArray::new(Buffer::from(sizes), Validity::NonNullable).into_array();
    ListViewArray::try_new(elements, offsets, sizes, validity)
}

/// Builds a canonical map whose rows may share entries.
pub(crate) fn build_map(
    map_dtype: &MapDType,
    keys: ArrayRef,
    values: ArrayRef,
    offsets: Vec<u64>,
    sizes: Vec<u64>,
    validity: Validity,
) -> VortexResult<MapArray> {
    let n = keys.len();
    let elements = StructArray::try_new(
        FieldNames::from(["key", "value"]),
        [keys, values],
        n,
        Validity::NonNullable,
    )?
    .into_array();
    MapArray::try_new(
        map_dtype.clone(),
        build_listview(elements, offsets, sizes, validity)?,
    )
}

/// Builds a non-nullable UTF-8 array from views that point into `buffers`.
pub(crate) fn utf8_from_views(views: Vec<BinaryView>, buffers: Arc<[ByteBuffer]>) -> ArrayRef {
    // SAFETY: the views are copied from arrays whose buffers are passed in the same positions, or
    // built from valid UTF-8 key bytes stored in the trailing pool buffer.
    unsafe {
        VarBinViewArray::new_unchecked(
            Buffer::from(views),
            buffers,
            DType::Utf8(Nullability::NonNullable),
            Validity::NonNullable,
        )
    }
    .into_array()
}
