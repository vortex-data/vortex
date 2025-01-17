use vortex_array::aliases::hash_set::HashSet;
use vortex_dtype::{DType, Field, FieldPath};
use vortex_error::{vortex_bail, VortexResult};

use crate::traversal::{FoldUp, Folder, Node};
use crate::{ExprRef, GetItem, Identity, Select};

/// Returns the field mask for the given expression.
///
/// This defines a mask over the scope of the fields that are accessed by the expression.
pub fn field_mask(expr: &ExprRef, scope_dtype: &DType) -> VortexResult<HashSet<FieldPath>> {
    // I know it's unused now, but we will for sure need the scope DType for future expressions.
    let DType::Struct(_scope_dtype, _) = scope_dtype else {
        vortex_bail!("Mismatched dtype {} for struct layout", scope_dtype);
    };

    Ok(match expr.accept_with_context(&mut FieldMaskFolder, ())? {
        FoldUp::Abort(out) => out,
        FoldUp::Continue(out) => out,
    })
}

struct FieldMaskFolder;

impl<'a> Folder<'a> for FieldMaskFolder {
    type NodeTy = ExprRef;
    type Out = HashSet<FieldPath>;
    type Context = ();

    fn visit_up(
        &mut self,
        node: &'a Self::NodeTy,
        _context: Self::Context,
        children: Vec<Self::Out>,
    ) -> VortexResult<FoldUp<Self::Out>> {
        // The identity returns a field path covering the root.
        if node.as_any().downcast_ref::<Identity>().is_some() {
            return Ok(FoldUp::Continue([FieldPath::root()].into()));
        }

        // GetItem pushes an element to each field path
        if let Some(getitem) = node.as_any().downcast_ref::<GetItem>() {
            let fields = children
                .into_iter()
                .flat_map(|field_mask| field_mask.into_iter())
                .map(|mut field_path| {
                    field_path.push(Field::Name(getitem.field().clone()));
                    field_path
                })
                .collect();
            return Ok(FoldUp::Continue(fields));
        }

        if node.as_any().downcast_ref::<Select>().is_some() {
            vortex_bail!("Expression must be simplified")
        }

        // Otherwise, return the field paths from the children
        Ok(FoldUp::Continue(children.into_iter().flatten().collect()))
    }
}
