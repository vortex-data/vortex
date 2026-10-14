// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A lookup array of UTF-8 or binary values, found through hash tables.

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::BuildHasher;
use std::hash::Hasher;
use std::sync::Arc;

use smallvec::smallvec;
use vortex_buffer::BitBuffer;
use vortex_compute::lane_kernels::collect_bits;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;
use vortex_utils::aliases::hash_map::HashTable;
use vortex_utils::aliases::hash_map::HashTableEntry;
use vortex_utils::aliases::hash_map::RandomState;

use super::HaystackOps;
use super::HaystackVTable;
use super::haystack_vtable_common;
use super::slot;
use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;
use crate::dtype::DType;
use crate::validity::Validity;

/// Builds the bytes haystack of the non-null `members`.
pub(super) fn try_new(
    members: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Array<BytesHaystack>> {
    let elements = members.execute::<VarBinViewArray>(ctx)?;
    let tables = BytesTables::new(&elements);
    let elements = elements.into_array();

    Array::try_from_parts(ArrayParts::new(
        BytesHaystack,
        elements.dtype().clone(),
        elements.len(),
        tables,
        smallvec![Some(elements)],
    ))
}

/// A lookup array of UTF-8 or binary values, found by their views where the view decides.
///
/// Its one child holds the elements, which may repeat. Its hash tables are its in-memory data.
///
/// A value of at most 12 bytes is inlined whole in its view, zero-padded as the compare kernel
/// requires, so a short needle is an element exactly when its whole view is the view of a short
/// element: no data buffer is read and no bytes are hashed. A long needle is found through a
/// table of the long elements, hashed by their bytes. The first 8 bytes of its view, the length
/// and a 4-byte prefix, are compared before the bytes after the prefix.
///
/// A one-bit filter over the heads of the elements was measured in front of these lookups and
/// removed: it helped short needles by about a tenth and cost long needles that share a prefix,
/// whose heads collide, up to a quarter.
#[derive(Clone, Debug)]
pub(crate) struct BytesHaystack;

impl BytesHaystack {
    const ELEMENTS: usize = 0;
}

/// The hash tables of a [`BytesHaystack`], over the views of its elements.
#[derive(Clone)]
pub(crate) struct BytesTables(Arc<Tables>);

struct Tables {
    hasher: RandomState,
    /// The distinct whole views of the elements of at most 12 bytes.
    short: HashTable<u128>,
    /// The indices of the distinct long elements, hashed by their bytes.
    long: HashTable<u32>,
}

impl BytesTables {
    fn new(elements: &VarBinViewArray) -> Self {
        let hasher = RandomState::default();
        let views = elements.views();
        let buffers = data_buffers(elements);

        // Sized up front, so that no insert grows a table and hashes every element again.
        let short_len = views.iter().filter(|view| view.is_inlined()).count();
        let mut short = HashTable::with_capacity(short_len);
        let mut long = HashTable::with_capacity(views.len() - short_len);

        for (idx, view) in views.iter().enumerate() {
            if view.is_inlined() {
                let whole = view.as_u128();
                if let HashTableEntry::Vacant(vacant) = short.entry(
                    hasher.hash_one(whole),
                    |&other| other == whole,
                    |&other| hasher.hash_one(other),
                ) {
                    vacant.insert(whole);
                }
                continue;
            }

            let value = view.bytes(&buffers);
            let bytes = |other: u32| views[other as usize].bytes(&buffers);
            if let HashTableEntry::Vacant(vacant) = long.entry(
                hasher.hash_one(value),
                |&other| bytes(other) == value,
                |&other| hasher.hash_one(bytes(other)),
            ) {
                vacant.insert(
                    u32::try_from(idx).vortex_expect("a list holds fewer than 2^32 elements"),
                );
            }
        }

        Self(Arc::new(Tables {
            hasher,
            short,
            long,
        }))
    }

    /// Whether `view`, which points into `buffers`, is the view of an element.
    #[inline]
    fn contains_view(
        &self,
        view: &BinaryView,
        buffers: &[&[u8]],
        element_views: &[BinaryView],
        element_buffers: &[&[u8]],
    ) -> bool {
        let tables = &self.0;
        if view.is_inlined() {
            let whole = view.as_u128();
            return tables
                .short
                .find(tables.hasher.hash_one(whole), |&other| other == whole)
                .is_some();
        }

        // Once the heads are equal, the bytes can differ only after the 4-byte prefix.
        let head = view_head(view);
        let value = view.bytes(buffers);
        tables
            .long
            .find(tables.hasher.hash_one(value), |&idx| {
                let element = &element_views[idx as usize];
                view_head(element) == head && element.bytes(element_buffers)[4..] == value[4..]
            })
            .is_some()
    }
}

impl Debug for BytesTables {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BytesTables")
            .field("short", &self.0.short.len())
            .field("long", &self.0.long.len())
            .finish()
    }
}

impl Display for BytesTables {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "short: {}, long: {}",
            self.0.short.len(),
            self.0.long.len()
        )
    }
}

// The tables follow from the elements, which the array compares and hashes as its child.
impl ArrayEq for BytesTables {
    fn array_eq(&self, _other: &Self, _accuracy: EqMode) -> bool {
        true
    }
}

impl ArrayHash for BytesTables {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _accuracy: EqMode) {}
}

impl VTable for BytesHaystack {
    type TypedArrayData = BytesTables;
    type OperationsVTable = HaystackOps;
    type ValidityVTable = HaystackOps;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.haystack.bytes");
        *ID
    }

    fn validate(
        &self,
        _data: &BytesTables,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure_eq!(slots.len(), 1, "A bytes haystack has one child");
        let elements = slots[Self::ELEMENTS]
            .as_ref()
            .ok_or_else(|| vortex_err!("A bytes haystack needs its elements"))?;
        vortex_ensure!(
            elements.is::<crate::arrays::VarBinView>(),
            "A bytes haystack indexes the views of a VarBinView array"
        );
        vortex_ensure_eq!(
            elements.dtype(),
            dtype,
            "A bytes haystack has its elements' dtype"
        );
        vortex_ensure_eq!(
            elements.len(),
            len,
            "A bytes haystack has a row per element"
        );
        Ok(())
    }

    haystack_vtable_common!("BytesHaystack");

    fn slot_name(_array: ArrayView<'_, Self>, _idx: usize) -> String {
        "elements".to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(
            slot(array.slots(), Self::ELEMENTS).clone(),
        ))
    }
}

impl HaystackVTable for BytesHaystack {
    fn contains(
        array: ArrayView<'_, Self>,
        needles: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<(BitBuffer, Validity)> {
        let elements = slot(array.slots(), Self::ELEMENTS)
            .clone()
            .execute::<VarBinViewArray>(ctx)?;
        let element_views = elements.views();
        let element_buffers = data_buffers(&elements);

        let needles = needles.clone().execute::<VarBinViewArray>(ctx)?;
        let buffers = data_buffers(&needles);
        let tables = array.data();
        let bits = collect_bits(
            needles.views(),
            |view: BinaryView| {
                tables.contains_view(&view, &buffers, element_views, &element_buffers)
            },
            ctx.allocator(),
        );
        Ok((bits, needles.validity()?))
    }
}

/// The head of a view: the `u32` length of its value and the first 4 bytes of the value,
/// zero-padded for a value shorter than 4 bytes.
#[inline]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the head is the low 8 bytes"
)]
fn view_head(view: &BinaryView) -> u64 {
    view.as_u128() as u64
}

/// The host slices of an array's data buffers, indexed by a view's buffer index.
fn data_buffers(array: &VarBinViewArray) -> Vec<&[u8]> {
    (0..array.data_buffers().len())
        .map(|idx| array.buffer(idx).as_slice())
        .collect()
}
