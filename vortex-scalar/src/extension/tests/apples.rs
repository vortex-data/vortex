// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;

use vortex_dtype::DType;
use vortex_dtype::ExtDType;
use vortex_dtype::ExtID;
use vortex_dtype::Nullability;
use vortex_dtype::PType;
use vortex_dtype::extension::ExtDTypeVTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ScalarValue;
use crate::extension::ExtScalarVTable;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
struct ApplesVTable;

impl ExtDTypeVTable for ApplesVTable {
    // Let's just say this is the number of times we repeat the scalar (which can be any type).
    type Metadata = usize;

    fn id(&self) -> ExtID {
        ExtID::new_ref("apples")
    }

    fn validate_dtype(&self, _options: &Self::Metadata, storage_dtype: &DType) -> VortexResult<()> {
        match storage_dtype {
            DType::List(..) => Ok(()),
            _ => Err(vortex_error::vortex_err!(
                "Expected list dtype for apples extension"
            )),
        }
    }
}

struct BasketOfApples(Vec<Option<ScalarValue>>);
impl Display for BasketOfApples {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, item) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            match item {
                Some(value) => write!(f, "{}", value)?,
                None => write!(f, "null")?,
            }
        }
        write!(f, "]")
    }
}

impl ExtScalarVTable for ApplesVTable {
    type Value<'a> = BasketOfApples;

    fn unpack<'a>(
        &self,
        _metadata: &'a <Self as ExtDTypeVTable>::Metadata,
        storage_dtype: &'a DType,
        storage_value: &'a ScalarValue,
    ) -> Self::Value<'a> {
        assert!(
            matches!(storage_dtype, DType::List(..)),
            "Expected list dtype"
        );

        let elements = storage_value.clone().into_list();
        BasketOfApples(elements)
    }

    fn validate_scalar_value(
        &self,
        _metadata: &<Self as ExtDTypeVTable>::Metadata,
        _storage_dtype: &DType,
        storage_value: &ScalarValue,
    ) -> VortexResult<()> {
        debug_assert!(
            matches!(storage_value, ScalarValue::List(_)),
            "Expected list variant"
        );

        // Since the dtype has been verified for this extension scalar, and list dtype must
        // always be paired with a `ScalarValue::List`, we don't have to do any checking here.
        Ok(())
    }
}

impl ApplesVTable {
    fn new() -> ExtDType<ApplesVTable> {
        ExtDType::try_new(0, DType::Primitive(PType::U16, Nullability::NonNullable))
            .vortex_expect("valid apples dtype")
    }
}
