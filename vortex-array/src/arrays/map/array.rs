// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hasher;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayRef;
use crate::ArraySlots;
use crate::EqMode;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::TypedArrayRef;
use crate::array_slots;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::map::Map;
use crate::dtype::DType;
use crate::dtype::MapDType;
use crate::validity::Validity;

#[array_slots(Map)]
pub struct MapSlots {
    /// The list-view storage of non-null `{key, value}` entry structs.
    #[slot(0)]
    pub entries: ArrayRef,
}

/// Encoding-specific metadata for [`crate::arrays::MapArray`].
///
/// All map metadata is represented by the outer [`DType::Map`] and the entries child, so this
/// value is intentionally empty.
#[derive(Clone, Debug, Default)]
pub struct MapData;

impl Display for MapData {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl ArrayEq for MapData {
    fn array_eq(&self, _other: &Self, _accuracy: EqMode) -> bool {
        true
    }
}

impl ArrayHash for MapData {
    fn array_hash<H: Hasher>(&self, _state: &mut H, _accuracy: EqMode) {}
}

impl MapData {
    pub(crate) fn make_slots(entries: ArrayRef) -> ArraySlots {
        MapSlots { entries }.into_slots()
    }
}

/// The logical and physical inputs used to construct a [`crate::arrays::MapArray`].
pub struct MapDataParts {
    /// The key/value type and sortedness assertion for the map.
    pub map_dtype: MapDType,
    /// The physical list-view storage of `{key, value}` entry structs.
    pub entries: ListViewArray,
}

/// Accessors for the canonical map representation.
pub trait MapArrayExt: MapArraySlotsExt {
    /// Returns the entry structs for one map row.
    fn entries_at(&self, index: usize) -> VortexResult<ArrayRef> {
        self.entries().as_::<ListView>().list_elements_at(index)
    }

    /// Returns the number of entries in one map row.
    fn entry_count_at(&self, index: usize) -> usize {
        self.entries().as_::<ListView>().size_at(index)
    }

    /// Returns the outer map validity delegated from the entries list-view.
    fn map_validity(&self) -> Validity {
        self.entries().as_::<ListView>().listview_validity()
    }

    /// Returns this map's key/value type information.
    fn map_dtype(&self) -> &MapDType {
        self.as_ref()
            .dtype()
            .as_map_opt()
            .vortex_expect("MapArray requires a map dtype")
    }

    /// Returns whether producers assert sorted keys within each map value.
    fn keys_sorted(&self) -> bool {
        self.map_dtype().keys_sorted()
    }
}
impl<T: TypedArrayRef<Map>> MapArrayExt for T {}

impl Array<Map> {
    /// Creates a canonical map array from its map dtype and list-view entry storage.
    ///
    /// # Panics
    ///
    /// Panics if `entries` is not a list of the map dtype's non-nullable `{key, value}` entry
    /// struct with matching outer nullability.
    pub fn new(map_dtype: MapDType, entries: ListViewArray) -> Self {
        Self::try_new(map_dtype, entries).vortex_expect("MapArray construction failed")
    }

    /// Constructs a canonical map array from its map dtype and list-view entry storage.
    ///
    /// # Errors
    ///
    /// Returns an error when the entry child is not `ListView<Struct<key, value>>`, has a
    /// different outer nullability, or has a different length than the outer map array.
    pub fn try_new(map_dtype: MapDType, entries: ListViewArray) -> VortexResult<Self> {
        let nullability = entries.nullability();
        let dtype = DType::Map(map_dtype, nullability);
        let len = entries.len();
        let slots = MapData::make_slots(entries.into_array());
        let parts = ArrayParts::new(Map, dtype, len, MapData).with_slots(slots);
        Self::try_from_parts(parts)
    }

    /// Creates a canonical map array without validating its entry storage.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `entries` has dtype
    /// `List(Struct { key, value }, entries.nullability())`, where the struct exactly matches
    /// `map_dtype.entries_dtype()`.
    pub unsafe fn new_unchecked(map_dtype: MapDType, entries: ListViewArray) -> Self {
        let nullability = entries.nullability();
        let dtype = DType::Map(map_dtype, nullability);
        let len = entries.len();
        let slots = MapData::make_slots(entries.into_array());
        let parts = ArrayParts::new(Map, dtype, len, MapData).with_slots(slots);
        unsafe { Self::from_parts_unchecked(parts) }
    }

    /// Decomposes this map array into its logical dtype and physical entries child.
    pub fn into_data_parts(self) -> MapDataParts {
        let map_dtype = self
            .dtype()
            .as_map_opt()
            .vortex_expect("MapArray requires a map dtype")
            .clone();
        let entries = self.entries().clone().downcast::<ListView>();
        MapDataParts { map_dtype, entries }
    }
}

fn expected_entries_dtype(map_dtype: &MapDType, nullability: crate::dtype::Nullability) -> DType {
    DType::List(Arc::new(map_dtype.entries_dtype()), nullability)
}

pub(super) fn validate_entries(
    map_dtype: &MapDType,
    nullability: crate::dtype::Nullability,
    len: usize,
    entries: &ArrayRef,
) -> VortexResult<()> {
    vortex_ensure!(
        entries.is::<ListView>(),
        "MapArray entries must use vortex.listview encoding, got {}",
        entries.encoding_id()
    );
    vortex_ensure!(
        entries.len() == len,
        "MapArray entries length {} does not match outer length {len}",
        entries.len()
    );

    let expected_dtype = expected_entries_dtype(map_dtype, nullability);
    vortex_ensure!(
        entries.dtype() == &expected_dtype,
        "MapArray entries dtype {} does not match expected {expected_dtype}",
        entries.dtype()
    );

    Ok(())
}
