use vortex_array::aliases::hash_set::HashSet;
use vortex_dtype::{DType, Field, FieldPath};
use vortex_error::{VortexResult, vortex_bail};

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
        if node.as_any().is::<Identity>() {
            return Ok(FoldUp::Continue([FieldPath::root()].into()));
        }

        // GetItem pushes an element to each field path
        if let Some(getitem) = node.as_any().downcast_ref::<GetItem>() {
            let fields = children
                .into_iter()
                .flat_map(|field_mask| field_mask.into_iter())
                .map(|field_path| field_path.push(Field::Name(getitem.field().clone())))
                .collect();
            return Ok(FoldUp::Continue(fields));
        }

        if node.as_any().is::<Select>() {
            vortex_bail!("Expression must be simplified")
        }

        // Otherwise, return the field paths from the children
        Ok(FoldUp::Continue(children.into_iter().flatten().collect()))
    }
}

#[cfg(test)]
mod test {
    use std::iter;
    use std::sync::Arc;

    use itertools::Itertools;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::{DType, FieldPath, PType, StructDType};

    use crate::transform::field_mask::field_mask;
    use crate::{get_item, ident};

    fn dtype() -> DType {
        DType::Struct(
            Arc::new(StructDType::new(
                ["A".into(), "B".into(), "C".into()].into(),
                iter::repeat_n(DType::Primitive(PType::I32, NonNullable), 3).collect(),
            )),
            NonNullable,
        )
    }

    #[test]
    fn field_mask_ident() {
        let mask = field_mask(&ident(), &dtype())
            .unwrap()
            .into_iter()
            .collect_vec();
        assert_eq!(mask.as_slice(), &[FieldPath::root()]);
    }

    #[test]
    fn field_mask_get_item() {
        let mask = field_mask(&get_item("A", ident()), &dtype())
            .unwrap()
            .into_iter()
            .collect_vec();
        assert_eq!(mask.as_slice(), &[FieldPath::from_name("A")]);
    }

    #[test]
    fn field_mask_get_item_nested() {
        let mask = field_mask(&get_item("B", get_item("A", ident())), &dtype())
            .unwrap()
            .into_iter()
            .collect_vec();
        assert_eq!(mask.as_slice(), &[FieldPath::from_name("A").push("B")]);
    }
}
