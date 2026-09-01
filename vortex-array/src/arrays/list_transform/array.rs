// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;

use crate::ArrayRef;
use crate::ArraySlots;
use crate::array::Array;
use crate::array::ArrayParts;
use crate::array::EmptyArrayData;
use crate::array::TypedArrayRef;
use crate::arrays::ListTransform;
use crate::arrays::list_transform::template::build_template;
use crate::dtype::DType;
use crate::expr::BoundLambda;

/// A lazy list transformation has structural children only:
/// `list`, a zero-length lambda `body`, then outer-row capture arrays.
pub trait ListTransformArrayExt: TypedArrayRef<ListTransform> {
    fn list(&self) -> &ArrayRef {
        self.as_ref().slots()[0]
            .as_ref()
            .vortex_expect("validated ListTransformArray list slot")
    }

    fn body(&self) -> &ArrayRef {
        self.as_ref().slots()[1]
            .as_ref()
            .vortex_expect("validated ListTransformArray body slot")
    }

    fn captures(&self) -> impl Iterator<Item = &ArrayRef> + '_ {
        self.as_ref().slots()[2..].iter().map(|capture| {
            capture
                .as_ref()
                .vortex_expect("validated ListTransformArray capture slot")
        })
    }

    fn capture_count(&self) -> usize {
        self.as_ref().slots().len() - 2
    }
}
impl<T: TypedArrayRef<ListTransform>> ListTransformArrayExt for T {}

impl Array<ListTransform> {
    /// Build a structural list transform from a bound lambda and outer-row captures.
    pub fn try_new(
        list: ArrayRef,
        lambda: BoundLambda,
        captures: impl IntoIterator<Item = ArrayRef>,
    ) -> VortexResult<Self> {
        let captures = captures.into_iter().collect::<Vec<_>>();
        validate_lambda(&list, &lambda, &captures)?;
        let body = build_template(&lambda)?;
        Self::try_new_from_parts(list, body, captures)
    }

    /// Rebuild a transform after substituting its outer template inputs.
    pub(crate) fn try_new_from_parts(
        list: ArrayRef,
        body: ArrayRef,
        captures: impl IntoIterator<Item = ArrayRef>,
    ) -> VortexResult<Self> {
        let captures = captures.into_iter().collect::<Vec<_>>();
        let dtype = output_dtype(list.dtype(), body.dtype())?;
        vortex_ensure!(
            body.is_empty(),
            "ListTransformArray body must be a zero-length template, got {}",
            body.len()
        );
        vortex_ensure!(
            captures.iter().all(|capture| capture.len() == list.len()),
            "ListTransformArray captures must have the outer list length {}",
            list.len()
        );
        let len = list.len();
        let slots = std::iter::once(list)
            .chain(std::iter::once(body))
            .chain(captures)
            .map(Some)
            .collect::<ArraySlots>();
        Array::try_from_parts(
            ArrayParts::new(ListTransform, dtype, len, EmptyArrayData).with_slots(slots),
        )
    }
}

pub(crate) fn output_dtype(list: &DType, body: &DType) -> VortexResult<DType> {
    match list {
        DType::List(_, nullability) => Ok(DType::List(body.clone().into(), *nullability)),
        DType::FixedSizeList(_, size, nullability) => Ok(DType::FixedSizeList(
            body.clone().into(),
            *size,
            *nullability,
        )),
        _ => vortex_bail!("list_transform() requires List, ListView, or FixedSizeList, got {list}"),
    }
}

fn validate_lambda(
    list: &ArrayRef,
    lambda: &BoundLambda,
    captures: &[ArrayRef],
) -> VortexResult<()> {
    let element_dtype = match list.dtype() {
        DType::List(element, _) | DType::FixedSizeList(element, ..) => element.as_ref(),
        _ => vortex_bail!(
            "list_transform() requires List, ListView, or FixedSizeList, got {}",
            list.dtype()
        ),
    };
    vortex_ensure!(
        matches!(lambda.param_dtypes().len(), 1 | 2),
        "list_transform() lambda must take one or two parameters, got {}",
        lambda.param_dtypes().len()
    );
    vortex_ensure!(
        lambda.param_dtypes()[0] == *element_dtype,
        "list_transform() element parameter expects dtype {}, got {}",
        lambda.param_dtypes()[0],
        element_dtype
    );
    if lambda.param_dtypes().len() == 2 {
        let index = DType::Primitive(
            crate::dtype::PType::U64,
            crate::dtype::Nullability::NonNullable,
        );
        vortex_ensure!(
            lambda.param_dtypes()[1] == index,
            "list_transform() index parameter expects dtype {index}, got {}",
            lambda.param_dtypes()[1]
        );
    }
    vortex_ensure!(
        lambda.captures().len() == captures.len(),
        "list_transform() lambda requires {} captures, got {}",
        lambda.captures().len(),
        captures.len()
    );
    for (index, (capture, array)) in lambda.captures().iter().zip(captures).enumerate() {
        vortex_ensure!(
            capture.dtype() == array.dtype(),
            "list_transform() capture {index} expects dtype {}, got {}",
            capture.dtype(),
            array.dtype()
        );
        vortex_ensure!(
            array.len() == list.len(),
            "list_transform() capture {index} has length {}, expected {}",
            array.len(),
            list.len()
        );
    }
    vortex_ensure!(
        lambda.body().is_root_bound_to(element_dtype),
        "list_transform() lambda root expects a different dtype than {element_dtype}"
    );
    Ok(())
}
