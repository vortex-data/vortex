// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ListArrowEncoder`] — short-circuits List → Arrow [`arrow_array::GenericListArray`] for
//! offset-based targets.

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::array::ArrayId;
use crate::array::ArrayPlugin;
use crate::arrays::List;
use crate::arrays::list::ListArrayExt;
use crate::arrow::ArrowEncoder;
use crate::arrow::ArrowSession;
use crate::arrow::executor::list::to_arrow_list;
use crate::dtype::PType;

/// Forward [`ArrowEncoder`] keyed by the [`crate::arrays::List`] [`ArrayId`].
///
/// Handles [`DataType::List`] and [`DataType::LargeList`] targets directly without canonicalizing
/// to [`crate::arrays::ListView`]. Returns [`None`] for any other target.
#[derive(Debug, Default)]
pub struct ListArrowEncoder;

impl ListArrowEncoder {
    /// The encoding [`ArrayId`] this encoder is registered against.
    pub fn array_id() -> ArrayId {
        List.id()
    }
}

impl ArrowEncoder for ListArrowEncoder {
    fn preferred_arrow_type(
        &self,
        array: &ArrayRef,
        session: &ArrowSession,
    ) -> VortexResult<Option<DataType>> {
        let Some(list) = array.as_opt::<List>() else {
            return Ok(None);
        };
        let offsets_ptype = PType::try_from(list.offsets().dtype())?;
        let use_large = matches!(offsets_ptype, PType::I64 | PType::U64);
        // Recurse via the session so nested List/VarBin children pick up their own
        // encoder-specific preferences.
        let elem_dtype = session.resolve_preferred_arrow_type(list.elements())?;
        let field = arrow_schema::FieldRef::new(arrow_schema::Field::new_list_field(
            elem_dtype,
            list.elements().dtype().is_nullable(),
        ));
        Ok(Some(if use_large {
            DataType::LargeList(field)
        } else {
            DataType::List(field)
        }))
    }

    fn to_arrow_array(
        &self,
        array: ArrayRef,
        target: &DataType,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrowArrayRef>> {
        match target {
            DataType::List(elements_field) => {
                to_arrow_list::<i32>(array, elements_field, ctx).map(Some)
            }
            DataType::LargeList(elements_field) => {
                to_arrow_list::<i64>(array, elements_field, ctx).map(Some)
            }
            _ => Ok(None),
        }
    }
}
