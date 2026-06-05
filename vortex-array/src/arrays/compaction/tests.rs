// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::Dict;
use crate::arrays::DictArray;
use crate::arrays::ListView;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::Struct;
use crate::arrays::StructArray;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::arrays::struct_::StructArrayExt;
use crate::assert_arrays_eq;
use crate::dtype::FieldNames;
use crate::validity::Validity;

#[test]
fn compact_dict_garbage_collects_dead_values() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();

    // Codes only reference values 0 and 2, so values 1 and 3 are dead. There is reuse (4 codes,
    // 2 live values), so the dictionary should be garbage collected rather than flattened.
    let codes = PrimitiveArray::from_iter(vec![0u32, 2, 0, 2]).into_array();
    let values = PrimitiveArray::from_iter(vec![10i32, 20, 30, 40]).into_array();
    let dict = DictArray::try_new(codes, values)?.into_array();

    let compacted = dict.compact(&mut ctx)?;

    let result_dict = compacted
        .as_opt::<Dict>()
        .expect("compacted dictionary should remain a dictionary");
    assert_eq!(
        result_dict.values().len(),
        2,
        "dead values should be dropped"
    );
    assert!(result_dict.has_all_values_referenced());

    assert_arrays_eq!(
        compacted,
        PrimitiveArray::from_iter(vec![10i32, 30, 10, 30])
    );
    Ok(())
}

#[test]
fn compact_dict_flattens_without_compression() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();

    // Every code maps to a distinct value, so the dictionary provides no compression and should be
    // decoded to a flat canonical array.
    let codes = PrimitiveArray::from_iter(vec![0u32, 1, 2]).into_array();
    let values = PrimitiveArray::from_iter(vec![10i32, 20, 30]).into_array();
    let dict = DictArray::try_new(codes, values)?.into_array();

    let compacted = dict.compact(&mut ctx)?;

    assert!(
        compacted.as_opt::<Dict>().is_none(),
        "a non-compressing dictionary should be flattened"
    );
    assert!(compacted.is_canonical());
    assert_arrays_eq!(compacted, PrimitiveArray::from_iter(vec![10i32, 20, 30]));
    Ok(())
}

#[test]
fn compact_list_view_becomes_zero_copy_to_list() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();

    // Elements 0, 2 and 4 are unreferenced gaps, so this list view is not zero-copy to a list.
    let elements = PrimitiveArray::from_iter(vec![10i32, 20, 30, 40, 50]).into_array();
    let offsets = PrimitiveArray::from_iter(vec![1i32, 3]).into_array();
    let sizes = PrimitiveArray::from_iter(vec![1i32, 1]).into_array();
    let list_view =
        ListViewArray::try_new(elements, offsets, sizes, Validity::NonNullable)?.into_array();
    assert!(
        !list_view
            .clone()
            .downcast::<ListView>()
            .is_zero_copy_to_list()
    );

    let compacted = list_view.clone().compact(&mut ctx)?;

    let result = compacted.clone().downcast::<ListView>();
    assert!(
        result.is_zero_copy_to_list(),
        "compacted list view should be zero-copy to list"
    );
    assert_arrays_eq!(compacted, list_view);
    Ok(())
}

#[test]
fn compact_struct_recurses_into_fields() -> VortexResult<()> {
    let mut ctx = LEGACY_SESSION.create_execution_ctx();

    let codes = PrimitiveArray::from_iter(vec![0u32, 2, 0, 2]).into_array();
    let values = PrimitiveArray::from_iter(vec![10i32, 20, 30, 40]).into_array();
    let dict = DictArray::try_new(codes, values)?.into_array();

    let struct_array = StructArray::try_new(
        FieldNames::from(["dict_field"]),
        vec![dict],
        4,
        Validity::NonNullable,
    )?
    .into_array();

    let compacted = struct_array.clone().compact(&mut ctx)?;

    // The field's dead dictionary values should have been garbage collected in place.
    let field = compacted.as_::<Struct>().unmasked_field(0).clone();
    let field_dict = field
        .as_opt::<Dict>()
        .expect("compacted struct field should remain a dictionary");
    assert_eq!(field_dict.values().len(), 2);
    assert!(field_dict.has_all_values_referenced());

    assert_arrays_eq!(compacted, struct_array);
    Ok(())
}
