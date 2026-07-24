// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Fuzz harness for the vortex-row byte encoder.
//!
//! Property: comparing two encoded row keys as byte strings yields the same ordering as a
//! logical tuple comparison of the row values under the configured [`RowSortField`]s. This
//! must hold across keys produced by *separate* encode calls over different chunks of the
//! same logical columns — chunks compress to different physical layouts (e.g. decimal value
//! widths), and sort/top-k operators compare keys across chunks.

use std::cmp::Ordering;

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::arbitrary::ArbitraryArray;
use vortex_array::arrays::arbitrary::ArbitraryArrayConfig;
use vortex_array::arrays::arbitrary::ArbitraryWith;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::arrays::listview::ListViewArrayExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::BigCast;
use vortex_array::dtype::DType;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::PType;
use vortex_array::dtype::half::f16;
use vortex_array::match_each_decimal_value_type;
use vortex_array::match_each_integer_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_row::RowSortField;
use vortex_row::compute_row_sizes;
use vortex_row::convert_columns;

use crate::SESSION;
use crate::array::CompressorStrategy;
use crate::array::compress_array;

/// Keep row counts small: the order property is asserted pairwise (O(rows²)).
const MAX_ROWS: usize = 96;

#[derive(Debug)]
pub struct FuzzRowEncode {
    /// Same-length columns of arbitrary dtypes.
    pub columns: Vec<ArrayRef>,
    /// Per-column sort configuration.
    pub fields: Vec<RowSortField>,
    /// Where to split the columns for the cross-chunk property.
    pub split_at: usize,
    /// Whether to btrblocks-compress each chunk before the cross-chunk encode.
    pub compress: bool,
}

impl<'a> Arbitrary<'a> for FuzzRowEncode {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let ncols = u.int_in_range(1..=3)?;
        let nrows = u.int_in_range(0..=MAX_ROWS)?;
        let mut columns = Vec::with_capacity(ncols);
        let mut fields = Vec::with_capacity(ncols);
        for _ in 0..ncols {
            let array = ArbitraryArray::arbitrary_with_config(
                u,
                &ArbitraryArrayConfig {
                    dtype: None,
                    len: nrows..=nrows,
                },
            )?
            .0;
            columns.push(array);
            fields.push(RowSortField::new(u.arbitrary()?, u.arbitrary()?));
        }
        Ok(Self {
            columns,
            fields,
            split_at: u.int_in_range(0..=nrows)?,
            compress: u.arbitrary()?,
        })
    }
}

/// Run the row-encoding properties for one generated input.
///
/// Returns `Ok(true)` to keep the input in the corpus, `Ok(false)` to reject inputs that the
/// encoder (correctly) refuses. Property violations panic.
pub fn run_row_encode(input: FuzzRowEncode) -> VortexResult<bool> {
    let mut ctx = SESSION.create_execution_ctx();
    let FuzzRowEncode {
        columns,
        fields,
        split_at,
        compress,
    } = input;

    if !columns.iter().all(|c| dtype_row_encodable(c.dtype())) {
        assert!(
            convert_columns(&columns, &fields, &mut ctx).is_err(),
            "row encoder accepted an unsupported dtype: {:?}",
            columns.iter().map(|c| c.dtype()).collect::<Vec<_>>()
        );
        return Ok(false);
    }

    let encoded = match convert_columns(&columns, &fields, &mut ctx) {
        Ok(encoded) => encoded,
        // Values that violate their declared decimal precision are rejected by the encoder;
        // such inputs are uninteresting, not property violations.
        Err(e) if e.to_string().contains("does not fit") => return Ok(false),
        Err(e) => return Err(e),
    };
    let rows = collect_row_bytes(&encoded, &mut ctx);
    let nrows = columns[0].len();

    // Property 1: per-row sizes match the encoded key lengths. `RowSize` reports a
    // `Struct { fixed, var }` per row; the total size is their sum.
    let sizes = compute_row_sizes(&columns, &fields, &mut ctx)?.execute::<StructArray>(&mut ctx)?;
    let parts = sizes
        .iter_unmasked_fields()
        .map(|f| f.clone().execute::<PrimitiveArray>(&mut ctx))
        .collect::<VortexResult<Vec<_>>>()?;
    for (i, row) in rows.iter().enumerate() {
        let total: u64 = parts.iter().map(|p| p.as_slice::<u32>()[i] as u64).sum();
        assert_eq!(
            total,
            row.len() as u64,
            "row {i}: computed size disagrees with encoded key length"
        );
    }

    // Property 2: byte order of whole-array keys equals the logical row order.
    let keys = columns
        .iter()
        .map(|c| row_keys(c, &mut ctx))
        .collect::<VortexResult<Vec<_>>>()?;
    let logical_cmp = |i: usize, j: usize| -> Ordering {
        fields
            .iter()
            .zip(&keys)
            .map(|(f, col)| col[i].compare(&col[j], *f))
            .find(|o| o.is_ne())
            .unwrap_or(Ordering::Equal)
    };
    for i in 0..nrows {
        for j in 0..nrows {
            assert_eq!(
                rows[i].cmp(&rows[j]),
                logical_cmp(i, j),
                "keys for rows {i} and {j} do not compare like their values \
                 (fields {fields:?}, dtypes {:?})",
                columns.iter().map(|c| c.dtype()).collect::<Vec<_>>()
            );
        }
    }

    // Property 3: keys stay comparable across separately encoded chunks, including chunks
    // whose physical layout was changed by compression.
    let chunk = |cols: &[ArrayRef], range: std::ops::Range<usize>, ctx: &mut ExecutionCtx| {
        cols.iter()
            .map(|c| {
                let sliced = c.slice(range.clone())?;
                Ok(if compress {
                    compress_array(&sliced, CompressorStrategy::Default, ctx)
                } else {
                    sliced
                })
            })
            .collect::<VortexResult<Vec<_>>>()
    };
    let head = chunk(&columns, 0..split_at, &mut ctx)?;
    let tail = chunk(&columns, split_at..nrows, &mut ctx)?;
    let head_rows = collect_row_bytes(&convert_columns(&head, &fields, &mut ctx)?, &mut ctx);
    let tail_rows = collect_row_bytes(&convert_columns(&tail, &fields, &mut ctx)?, &mut ctx);
    for (i, head_row) in head_rows.iter().enumerate() {
        for (j, tail_row) in tail_rows.iter().enumerate() {
            assert_eq!(
                head_row.cmp(tail_row),
                logical_cmp(i, split_at + j),
                "cross-chunk keys for rows {i} and {} do not compare like their values \
                 (split at {split_at}, compress {compress}, fields {fields:?}, dtypes {:?})",
                split_at + j,
                columns.iter().map(|c| c.dtype()).collect::<Vec<_>>()
            );
        }
    }

    Ok(true)
}

/// Dtypes the row encoder supports; everything else must be rejected with an error.
fn dtype_row_encodable(dtype: &DType) -> bool {
    match dtype {
        DType::Null | DType::Bool(_) | DType::Primitive(..) | DType::Utf8(_) | DType::Binary(_) => {
            true
        }
        DType::Decimal(dt, _) => !matches!(
            DecimalType::smallest_decimal_value_type(dt),
            DecimalType::I256
        ),
        DType::Struct(fields, _) => fields.fields().all(|f| dtype_row_encodable(&f)),
        DType::FixedSizeList(elem, ..) => dtype_row_encodable(elem),
        _ => false,
    }
}

fn collect_row_bytes(array: &ListViewArray, ctx: &mut ExecutionCtx) -> Vec<Vec<u8>> {
    (0..array.len())
        .map(|i| {
            let slice = array
                .list_elements_at(i)
                .vortex_expect("row key bytes must be addressable");
            let p = slice
                .execute::<PrimitiveArray>(ctx)
                .vortex_expect("row key bytes must canonicalize");
            p.as_slice::<u8>().to_vec()
        })
        .collect()
}

/// A row value lifted into a form whose comparison mirrors the byte encoding's semantics.
#[derive(Debug, Clone, PartialEq)]
enum RowKey {
    Null,
    Bool(bool),
    /// Every integer ptype and decimal unscaled value (Decimal256 is unsupported upstream).
    Int(i128),
    /// IEEE-754 total-order bit key (sign-flipped bits), matching the encoded byte order.
    Float(u64),
    Bytes(Vec<u8>),
    /// Struct fields or fixed-size-list elements, in order.
    Composite(Vec<RowKey>),
}

impl RowKey {
    /// Compare as the encoded bytes compare: nulls are positioned by `nulls_first` regardless
    /// of direction, and `descending` inverts leaf values only — the sentinel bytes that
    /// place nulls are never inverted, at any nesting depth.
    fn compare(&self, other: &Self, field: RowSortField) -> Ordering {
        use RowKey::*;
        match (self, other) {
            (Null, Null) => Ordering::Equal,
            (Null, _) => {
                if field.nulls_first {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            }
            (_, Null) => {
                if field.nulls_first {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            }
            (Composite(lhs), Composite(rhs)) => lhs
                .iter()
                .zip(rhs)
                .map(|(l, r)| l.compare(r, field))
                .find(|o| o.is_ne())
                .unwrap_or(Ordering::Equal),
            (lhs, rhs) => {
                let natural = match (lhs, rhs) {
                    (Bool(l), Bool(r)) => l.cmp(r),
                    (Int(l), Int(r)) => l.cmp(r),
                    (Float(l), Float(r)) => l.cmp(r),
                    (Bytes(l), Bytes(r)) => l.cmp(r),
                    _ => unreachable!("mismatched RowKey kinds within one column"),
                };
                if field.descending {
                    natural.reverse()
                } else {
                    natural
                }
            }
        }
    }
}

fn f64_key(v: f64) -> u64 {
    let bits = v.to_bits();
    if bits >> 63 == 1 {
        !bits
    } else {
        bits ^ (1 << 63)
    }
}

fn f32_key(v: f32) -> u64 {
    let bits = v.to_bits();
    (if bits >> 31 == 1 {
        !bits
    } else {
        bits ^ (1 << 31)
    }) as u64
}

fn f16_key(v: f16) -> u64 {
    let bits = v.to_bits();
    (if bits >> 15 == 1 {
        !bits
    } else {
        bits ^ (1 << 15)
    }) as u64
}

/// Lift every row of `array` into a [`RowKey`].
fn row_keys(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Vec<RowKey>> {
    let len = array.len();
    let with_mask = |keys: Vec<RowKey>, array: &ArrayRef, ctx: &mut ExecutionCtx| {
        let mask = array.validity()?.execute_mask(len, ctx)?;
        Ok(keys
            .into_iter()
            .enumerate()
            .map(|(i, k)| if mask.value(i) { k } else { RowKey::Null })
            .collect())
    };
    match array.clone().execute::<Canonical>(ctx)? {
        Canonical::Null(_) => Ok(vec![RowKey::Null; len]),
        Canonical::Bool(a) => {
            let bits = a.to_bit_buffer();
            let keys = (0..len).map(|i| RowKey::Bool(bits.value(i))).collect();
            with_mask(keys, &a.into_array(), ctx)
        }
        Canonical::Primitive(a) => {
            let keys = match a.ptype() {
                PType::F16 => a
                    .as_slice::<f16>()
                    .iter()
                    .map(|v| RowKey::Float(f16_key(*v)))
                    .collect(),
                PType::F32 => a
                    .as_slice::<f32>()
                    .iter()
                    .map(|v| RowKey::Float(f32_key(*v)))
                    .collect(),
                PType::F64 => a
                    .as_slice::<f64>()
                    .iter()
                    .map(|v| RowKey::Float(f64_key(*v)))
                    .collect(),
                _ => match_each_integer_ptype!(a.ptype(), |P| {
                    a.as_slice::<P>()
                        .iter()
                        .map(|v| RowKey::Int((*v).into()))
                        .collect::<Vec<_>>()
                }),
            };
            with_mask(keys, &a.into_array(), ctx)
        }
        Canonical::Decimal(a) => {
            let mask = a.as_ref().validity()?.execute_mask(len, ctx)?;
            match_each_decimal_value_type!(a.values_type(), |D| {
                let buf = a.buffer::<D>();
                Ok((0..len)
                    .map(|i| {
                        if mask.value(i) {
                            RowKey::Int(<i128 as BigCast>::from(buf[i]).vortex_expect(
                                "valid decimal values fit i128 (Decimal256 dtypes are filtered)",
                            ))
                        } else {
                            RowKey::Null
                        }
                    })
                    .collect())
            })
        }
        Canonical::VarBinView(a) => {
            let mask = a.as_ref().validity()?.execute_mask(len, ctx)?;
            Ok((0..len)
                .map(|i| {
                    if mask.value(i) {
                        RowKey::Bytes(a.bytes_at(i).to_vec())
                    } else {
                        RowKey::Null
                    }
                })
                .collect())
        }
        Canonical::Struct(a) => {
            let children = a
                .iter_unmasked_fields()
                .map(|child| row_keys(child, ctx))
                .collect::<VortexResult<Vec<_>>>()?;
            let keys = (0..len)
                .map(|i| RowKey::Composite(children.iter().map(|c| c[i].clone()).collect()))
                .collect();
            with_mask(keys, &a.into_array(), ctx)
        }
        Canonical::FixedSizeList(a) => {
            let k = a.list_size() as usize;
            let elements = row_keys(&a.elements().clone(), ctx)?;
            let keys = (0..len)
                .map(|i| RowKey::Composite(elements[i * k..(i + 1) * k].to_vec()))
                .collect();
            with_mask(keys, &a.into_array(), ctx)
        }
        c => unreachable!(
            "unsupported dtypes are rejected before oracle construction: {:?}",
            c.dtype()
        ),
    }
}
