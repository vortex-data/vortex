// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Detects rows of a column that equal the previous row.
//!
//! Label maps are usually sorted by series, so consecutive rows repeat the same labels. Decoding
//! points such rows at the previous row's entries instead of materializing them again.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::Dict;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::UnionArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::arrays::union::UnionArrayExt;
use vortex_array::arrays::union::UnionArraySlotsExt;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::match_each_integer_ptype;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_sparse::Sparse;
use vortex_sparse::SparseExt;

use crate::flat::to_usize_vec;

enum Values {
    Views {
        array: VarBinViewArray,
        buffers: Vec<ByteBuffer>,
    },
    Bytes {
        bytes: ByteBuffer,
        width: usize,
    },
    Ints(PrimitiveArray),
    Bool(BitBuffer),
    Union {
        tags: ByteBuffer,
        child_of_tag: Box<[usize; 256]>,
        children: Vec<RowCmp>,
    },
    /// A sparse column: ascending present rows and a comparator over the patch values.
    Sparse {
        rows: Vec<usize>,
        values: Box<RowCmp>,
    },
    /// No cheap comparison: every row counts as changed.
    Unknown,
}

/// Compares row `i` of a canonical column with row `i - 1`.
pub(crate) struct RowCmp {
    validity: Option<Mask>,
    values: Values,
}

impl RowCmp {
    pub fn new(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        // Equal codes mean equal values. Unequal codes may still hold equal values, which only
        // costs a missed repeat.
        if let Some(dict) = array.as_opt::<Dict>() {
            return Self::new(dict.codes(), ctx);
        }
        if let Some(sparse) = array.as_opt::<Sparse>()
            && sparse.patches().offset() == 0
            && sparse.fill_scalar().is_null()
        {
            let patches = sparse.patches();
            return Ok(Self {
                validity: None,
                values: Values::Sparse {
                    rows: to_usize_vec(patches.indices(), ctx)?,
                    values: Box::new(Self::new(patches.values(), ctx)?),
                },
            });
        }
        let validity = if array.dtype().is_nullable() {
            Some(array.validity()?.execute_mask(array.len(), ctx)?)
        } else {
            None
        };
        let values = match array.clone().execute::<Canonical>(ctx)? {
            Canonical::VarBinView(a) => views_of(&a),
            Canonical::Primitive(a) if a.ptype().is_int() => Values::Ints(a),
            Canonical::Primitive(a) => Values::Bytes {
                width: a.ptype().byte_width(),
                bytes: a.buffer_handle().as_host().clone(),
            },
            Canonical::Bool(a) => Values::Bool(a.to_bit_buffer()),
            Canonical::Union(a) => union_of(&a, ctx)?,
            _ => Values::Unknown,
        };
        Ok(Self { validity, values })
    }

    /// Whether rows `left` and `right` hold equal values, treating two nulls as equal.
    #[inline]
    pub fn equal(&self, left: usize, right: usize) -> bool {
        if let Some(mask) = &self.validity {
            let (left_valid, right_valid) = (mask.value(left), mask.value(right));
            if left_valid != right_valid {
                return false;
            }
            if !left_valid {
                return true;
            }
        }
        self.values_equal(left, right)
    }

    /// For every row, whether it equals the previous row. Row 0 is never set.
    pub fn same_as_prev_bits(&self, len: usize) -> BitBuffer {
        if let Values::Sparse { rows, values } = &self.values {
            // Absent rows equal absent rows; a present row equals a present predecessor with an
            // equal value at the previous rank.
            let mut present = BitBufferMut::new_unset(len);
            for &row in rows {
                present.set(row);
            }
            let present = present.freeze();
            let mut same = BitBufferMut::collect_bool(len, |left| {
                left > 0 && !present.value(left) && !present.value(left - 1)
            });
            for (rank, &row) in rows.iter().enumerate() {
                if rank > 0 && row > 0 && rows[rank - 1] == row - 1 && values.equal(rank, rank - 1)
                {
                    same.set(row);
                }
            }
            return same.freeze();
        }
        let values = match &self.values {
            Values::Ints(ints) => match_each_integer_ptype!(ints.ptype(), |P| {
                let values = ints.as_slice::<P>();
                BitBuffer::collect_bool(len, |left| left > 0 && values[left] == values[left - 1])
            }),
            _ => BitBuffer::collect_bool(len, |left| left > 0 && self.values_equal(left, left - 1)),
        };
        let Some(mask) = &self.validity else {
            return values;
        };
        match mask.bit_buffer() {
            AllOr::All => values,
            AllOr::None => BitBuffer::collect_bool(len, |left| left > 0),
            AllOr::Some(valid) => BitBuffer::collect_bool(len, |left| {
                left > 0
                    && valid.value(left) == valid.value(left - 1)
                    && (!valid.value(left) || values.value(left))
            }),
        }
    }

    fn values_equal(&self, left: usize, right: usize) -> bool {
        match &self.values {
            Values::Views { array, buffers } => {
                let views: &[BinaryView] = array.views();
                let (lhs, rhs) = (views[left], views[right]);
                if lhs.as_u128() == rhs.as_u128() {
                    return true;
                }
                if lhs.len() != rhs.len() || lhs.is_inlined() {
                    return false;
                }
                let (lhs, rhs) = (lhs.as_view(), rhs.as_view());
                buffers[lhs.buffer_index as usize][lhs.as_range()]
                    == buffers[rhs.buffer_index as usize][rhs.as_range()]
            }
            Values::Bytes { bytes, width } => {
                bytes[left * width..(left + 1) * width] == bytes[right * width..(right + 1) * width]
            }
            Values::Ints(ints) => match_each_integer_ptype!(ints.ptype(), |P| {
                let values = ints.as_slice::<P>();
                values[left] == values[right]
            }),
            Values::Bool(bits) => bits.value(left) == bits.value(right),
            Values::Union {
                tags,
                child_of_tag,
                children,
            } => {
                tags[left] == tags[right] && {
                    let child = &children[child_of_tag[tags[left] as usize]];
                    match &child.validity {
                        Some(mask) if mask.value(left) != mask.value(right) => false,
                        Some(mask) if !mask.value(left) => true,
                        _ => child.values_equal(left, right),
                    }
                }
            }
            Values::Sparse { rows, values } => {
                match (rows.binary_search(&left), rows.binary_search(&right)) {
                    (Ok(lhs), Ok(rhs)) => values.equal(lhs, rhs),
                    (Err(_), Err(_)) => true,
                    _ => false,
                }
            }
            Values::Unknown => false,
        }
    }
}

fn views_of(array: &VarBinViewArray) -> Values {
    Values::Views {
        array: array.clone(),
        buffers: array
            .data_buffers()
            .iter()
            .map(|b| b.as_host().clone())
            .collect(),
    }
}

fn union_of(array: &UnionArray, ctx: &mut ExecutionCtx) -> VortexResult<Values> {
    let tags = array.type_ids().clone().execute::<PrimitiveArray>(ctx)?;
    let mut child_of_tag = Box::new([0usize; 256]);
    for (child, &tag) in array.variants().type_ids().iter().enumerate() {
        child_of_tag[tag as usize] = child;
    }
    let children = array
        .iter_children()
        .map(|c| RowCmp::new(c, ctx))
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(Values::Union {
        tags: tags.buffer_handle().as_host().clone(),
        child_of_tag,
        children,
    })
}
