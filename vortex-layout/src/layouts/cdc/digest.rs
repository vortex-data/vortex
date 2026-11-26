// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reduce a canonical chunk to one digest per row.
//!
//! [`row_digests`] walks a canonical array and returns a [`RowDigest`] for every row: a 64-bit
//! digest of the row's logical content, and the number of bytes that content would occupy
//! serialized. [`RollingCutter`](super::RollingCutter) rolls the digests through its GEAR hash to
//! place chunk boundaries, and spends the widths as its chunk size budget.
//!
//! Rows are reduced to digests rather than hashed from their encoded bytes because a digest must
//! depend only on the row's own logical values: identical rows then digest identically wherever
//! they appear, in this file or in a later version of it, which is what lets boundaries
//! re-synchronize after an edit shifts rows.
//!
//! Every digest is whitened through [`mix64`]. Typical columns (sequential ids, near-constant
//! timestamps, low-cardinality categories) carry very little per-byte entropy, and hashing their
//! bytes directly starves the rolling hash of boundary candidates, degrading cuts into fixed
//! strides that never re-align after a shift. Whitening restores a uniform candidate
//! distribution for any content while staying a pure function of the row.
//!
//! Null rows fold only their validity marker. The value behind a null is undefined padding that
//! need not be a function of the logical content, so letting it reach the digest would make
//! boundaries depend on how nulls happened to be encoded.

use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::decimal::DecimalArrayExt;
use vortex_array::arrays::extension::ExtensionArraySlotsExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArrayExt;
use vortex_array::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use vortex_array::arrays::listview::ListViewArrayExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::map::MapArraySlotsExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::arrays::union::UnionArrayExt;
use vortex_array::arrays::union::UnionArraySlotsExt;
use vortex_array::arrays::varbinview::VarBinViewArrayExt;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_decimal_value_type;
use vortex_array::match_each_integer_ptype;
use vortex_array::match_each_native_ptype;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

/// One row's digest and the serialized width of its content in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct RowDigest {
    pub(super) hash: u64,
    pub(super) width: u64,
}

impl RowDigest {
    /// Fold a nested or sibling value into this row's digest.
    #[inline]
    fn fold(&mut self, other: Self) {
        self.hash = mix64(self.hash ^ other.hash);
        self.width += other.width;
    }
}

/// A SplitMix64-style finalizer that whitens content before it reaches the rolling hash.
#[inline]
pub(super) fn mix64(value: u64) -> u64 {
    let mut z = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Fold a byte string into a digest, eight little-endian bytes at a time.
#[inline]
fn fold_bytes(mut digest: u64, bytes: &[u8]) -> u64 {
    let (words, tail) = bytes.as_chunks::<8>();
    for word in words {
        digest = mix64(digest ^ u64::from_le_bytes(*word));
    }
    if !tail.is_empty() {
        let mut word = [0u8; 8];
        word[..tail.len()].copy_from_slice(tail);
        digest = mix64(digest ^ u64::from_le_bytes(word));
    }
    digest
}

/// Digest every row of `canonical`.
pub(super) fn row_digests(
    canonical: &Canonical,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<RowDigest>> {
    let len = canonical.len();
    match canonical {
        // A null column has no content to distinguish its rows, so every row digests alike and
        // boundaries within it fall back to `max_chunk_bytes` strides. Rows are still charged a
        // byte apiece: a zero width would never advance the chunk budget, and the column would
        // never be cut at all.
        Canonical::Null(array) => Ok(vec![
            RowDigest {
                hash: mix64(0),
                width: 1
            };
            array.len()
        ]),

        Canonical::Bool(array) => {
            let validity = array.validity()?;
            let bits = array.clone().into_bit_buffer();
            leaf_digests(len, &validity, ctx, |row, seed| {
                (mix64(seed ^ u64::from(bits.value(row))), 1)
            })
        }

        Canonical::Primitive(array) => {
            let validity = PrimitiveArrayExt::validity(array);
            let width = array.ptype().byte_width();
            let bytes = match_each_native_ptype!(array.ptype(), |P| {
                array.to_buffer::<P>().into_byte_buffer()
            });
            let bytes = bytes.as_slice();
            leaf_digests(len, &validity, ctx, |row, seed| {
                (
                    fold_bytes(seed, &bytes[row * width..(row + 1) * width]),
                    width as u64,
                )
            })
        }

        Canonical::Decimal(array) => {
            let validity = DecimalArrayExt::validity(array);
            let width = array.values_type().byte_width();
            let bytes = match_each_decimal_value_type!(array.values_type(), |D| {
                array.buffer::<D>().into_byte_buffer()
            });
            let bytes = bytes.as_slice();
            leaf_digests(len, &validity, ctx, |row, seed| {
                (
                    fold_bytes(seed, &bytes[row * width..(row + 1) * width]),
                    width as u64,
                )
            })
        }

        Canonical::VarBinView(array) => {
            let validity = array.varbinview_validity();
            leaf_digests(len, &validity, ctx, |row, seed| {
                let view = &array.views()[row];
                let seed = mix64(seed ^ u64::from(view.len()));
                let digest = if view.is_inlined() {
                    fold_bytes(seed, view.as_inlined().value())
                } else {
                    let region = view.as_view();
                    fold_bytes(
                        seed,
                        &array.buffer(region.buffer_index as usize).as_slice()[region.as_range()],
                    )
                };
                (digest, 4 + u64::from(view.len()))
            })
        }

        Canonical::Struct(array) => {
            let validity = RowValidity::new(&array.struct_validity(), len, ctx)?;
            let mut digests = marker_digests(&validity, len);
            for field in array.iter_unmasked_fields() {
                let field = field.clone().execute::<Canonical>(ctx)?;
                let field = row_digests(&field, ctx)?;
                for (row, (digest, field)) in digests.iter_mut().zip(field).enumerate() {
                    // The fields behind a null struct are undefined, like any other null's.
                    if validity.is_valid(row) {
                        digest.fold(field);
                    }
                }
            }
            Ok(digests)
        }

        Canonical::FixedSizeList(array) => {
            let validity = array.fixed_size_list_validity();
            let list_size = usize::try_from(array.list_size())
                .map_err(|_| vortex_err!("fixed size list size overflows usize"))?;
            let elements = child_digests(array.elements(), ctx)?;
            nested_digests(len, &validity, ctx, &elements, |row| {
                row * list_size..(row + 1) * list_size
            })
        }

        Canonical::List(array) => {
            let validity = array.listview_validity();
            let elements = child_digests(array.elements(), ctx)?;
            // List views may overlap and need not ascend, so each row's range comes from its own
            // offset and size rather than from a cumulative scan of the offsets.
            let offsets = to_indices(array.offsets(), ctx)?;
            let sizes = to_indices(array.sizes(), ctx)?;
            nested_digests(len, &validity, ctx, &elements, |row| {
                offsets[row]..offsets[row] + sizes[row]
            })
        }

        // A map is stored as a list of non-nullable {key, value} structs, and that list also
        // carries the map's validity, so its rows already digest exactly as the map's rows do.
        Canonical::Map(array) => row_digests(&array.entries().clone().execute(ctx)?, ctx),

        Canonical::Union(array) => {
            let type_ids = array.type_ids().clone().execute::<PrimitiveArray>(ctx)?;
            // A union's nulls live in its type ids rather than in a validity slot.
            let validity = PrimitiveArrayExt::validity(&type_ids);
            let tags = type_ids.as_slice::<u8>();
            // Unions are sparse: every child is row-aligned with the union, so each child is
            // digested once and the tag selects which digest a row takes.
            let children = array
                .iter_children()
                .map(|child| child_digests(child, ctx))
                .collect::<VortexResult<Vec<_>>>()?;
            let variants = array.variants();
            leaf_digests(len, &validity, ctx, |row, seed| {
                let tag = tags[row];
                let seed = mix64(seed ^ u64::from(tag));
                match variants
                    .tag_to_child_index(tag)
                    .and_then(|child| children.get(child))
                {
                    Some(child) => (mix64(seed ^ child[row].hash), 1 + child[row].width),
                    None => (seed, 1),
                }
            })
        }

        Canonical::Extension(array) => row_digests(&array.storage().clone().execute(ctx)?, ctx),

        // A variant row's value spans an arbitrary encoding plus an optional shredded tree that
        // `execute_scalar` merges, so there is no flat buffer to walk: digest the scalars
        // themselves, at the cost of materializing one per row.
        Canonical::Variant(array) => {
            let validity = RowValidity::new(&array.validity()?, len, ctx)?;
            let mut digests = marker_digests(&validity, len);
            let array = array.clone().into_array();
            // This mirrors `leaf_digests`, which cannot be reused here because reading a scalar
            // borrows the execution context that its closure would have to capture.
            for (row, digest) in digests.iter_mut().enumerate() {
                if validity.is_valid(row) {
                    let mut hasher = DigestHasher {
                        digest: digest.hash,
                        width: 0,
                    };
                    array.execute_scalar(row, ctx)?.hash(&mut hasher);
                    digest.hash = hasher.digest;
                    digest.width += hasher.width;
                }
            }
            Ok(digests)
        }
    }
}

/// Digest a child array, canonicalizing it first.
fn child_digests(child: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Vec<RowDigest>> {
    let child = child.clone().execute::<Canonical>(ctx)?;
    row_digests(&child, ctx)
}

/// Digest the rows of a leaf array.
///
/// Each row is seeded with its validity marker; `value` folds a valid row's content into that
/// seed and reports the content's serialized width. Null rows keep the bare seed.
fn leaf_digests(
    len: usize,
    validity: &Validity,
    ctx: &mut ExecutionCtx,
    mut value: impl FnMut(usize, u64) -> (u64, u64),
) -> VortexResult<Vec<RowDigest>> {
    let validity = RowValidity::new(validity, len, ctx)?;
    let mut digests = marker_digests(&validity, len);
    for (row, digest) in digests.iter_mut().enumerate() {
        if validity.is_valid(row) {
            let (hash, width) = value(row, digest.hash);
            digest.hash = hash;
            digest.width += width;
        }
    }
    Ok(digests)
}

/// The validity marker alone: what a null row digests to, and the seed a valid row folds its
/// content into.
fn marker_digests(validity: &RowValidity, len: usize) -> Vec<RowDigest> {
    let width = validity.marker_width();
    (0..len)
        .map(|row| RowDigest {
            hash: mix64(u64::from(validity.is_valid(row))),
            width,
        })
        .collect()
}

/// Digest rows whose content is a range of a child array's already-digested elements.
fn nested_digests(
    len: usize,
    validity: &Validity,
    ctx: &mut ExecutionCtx,
    elements: &[RowDigest],
    mut range: impl FnMut(usize) -> Range<usize>,
) -> VortexResult<Vec<RowDigest>> {
    leaf_digests(len, validity, ctx, |row, seed| {
        let range = range(row);
        // The element count joins the digest so that `[1, 2]` and `[1, 2, 2]` cannot collide.
        let mut digest = RowDigest {
            hash: mix64(seed ^ range.len() as u64),
            width: 4,
        };
        for element in &elements[range] {
            digest.fold(*element);
        }
        (digest.hash, digest.width)
    })
}

/// Read an offsets or sizes array into element indices.
// The macro covers every integer width, including the signed and 64-bit ones whose conversion
// really can fail, so the conversion stays fallible for the narrow arms too.
#[allow(clippy::unnecessary_fallible_conversions)]
fn to_indices(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Vec<usize>> {
    let array = array.clone().execute::<PrimitiveArray>(ctx)?;
    match_each_integer_ptype!(array.ptype(), |P| {
        array
            .as_slice::<P>()
            .iter()
            .map(|index| {
                usize::try_from(*index).map_err(|_| vortex_err!("list index does not fit usize"))
            })
            .collect()
    })
}

/// Per-row validity, resolved once so that row loops avoid re-dispatching on the mask.
enum RowValidity {
    /// The dtype is non-nullable, so rows carry no validity marker at all.
    NonNullable,
    AllValid,
    AllNull,
    Bits(BitBuffer),
}

impl RowValidity {
    fn new(validity: &Validity, len: usize, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        if validity.nullability() == Nullability::NonNullable {
            return Ok(Self::NonNullable);
        }
        Ok(match validity.execute_mask(len, ctx)? {
            Mask::AllTrue(_) => Self::AllValid,
            Mask::AllFalse(_) => Self::AllNull,
            Mask::Values(values) => Self::Bits(values.bit_buffer().clone()),
        })
    }

    #[inline]
    fn is_valid(&self, row: usize) -> bool {
        match self {
            Self::NonNullable | Self::AllValid => true,
            Self::AllNull => false,
            Self::Bits(bits) => bits.value(row),
        }
    }

    /// The bytes a row's validity marker adds to its serialized width.
    #[inline]
    fn marker_width(&self) -> u64 {
        match self {
            Self::NonNullable => 0,
            _ => 1,
        }
    }
}

/// A [`Hasher`] that folds written bytes into a whitened digest, so values with no flat buffer
/// to walk can be digested through their [`Hash`] implementation.
struct DigestHasher {
    digest: u64,
    width: u64,
}

impl Hasher for DigestHasher {
    fn write(&mut self, bytes: &[u8]) {
        self.digest = fold_bytes(self.digest, bytes);
        self.width += bytes.len() as u64;
    }

    fn finish(&self) -> u64 {
        self.digest
    }
}
