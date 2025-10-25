// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::stream::SendableArrayStream;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;

pub trait ArraySource {
    fn produce(self: Box<Self>) -> VortexResult<SendableArrayStream>;
}

pub trait ArrayTransform {
    fn transform(self: Box<Self>, stream: SendableArrayStream)
    -> VortexResult<SendableArrayStream>;
}

pub trait ArraySink {
    fn consume(self: Box<Self>, stream: SendableArrayStream) -> VortexResult<()>;
}

pub struct ArrayStreamNode {
    array: ArrayRef,
}

#[cfg(test)]
mod test {
    use crate::layouts::flat::FlatLayout;
    use crate::layouts::struct_::StructLayout;
    use crate::segments::SegmentId;
    use crate::{IntoLayout, LayoutRef};
    use vortex_array::ArrayContext;
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::{DType, FieldNames, StructFields};
    use vortex_expr::{col, gt};

    fn create_test_layout() -> LayoutRef {
        let ctx = ArrayContext::empty();
        let col_a = FlatLayout::new(
            10,
            DType::Bool(NonNullable),
            SegmentId::from(0),
            ctx.clone(),
        )
        .into_layout();
        let col_b = FlatLayout::new(
            10,
            DType::Bool(NonNullable),
            SegmentId::from(1),
            ctx.clone(),
        )
        .into_layout();
        let fields = StructFields::new(
            FieldNames::from_iter(["a", "b"]),
            vec![col_a.dtype().clone(), col_b.dtype().clone()],
        );
        StructLayout::new(10, DType::Struct(fields, NonNullable), vec![col_a, col_b]).into_layout()
    }

    #[test]
    fn foo() {
        let layout = create_test_layout();

        // Now we create an expression.
        let expr = gt(col("a"), col("b"));
    }
}
