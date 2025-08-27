// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::{VortexError, VortexResult, vortex_err};

use crate::field::{Field, FieldPath};
use crate::proto::dtype as pb;
use crate::proto::dtype::d_type::DtypeType;
use crate::proto::dtype::field::FieldType;
use crate::{DType, DecimalDType, ExtDType, ExtID, ExtMetadata, PType, StructFields};

impl TryFrom<&pb::DType> for DType {
    type Error = VortexError;

    fn try_from(value: &pb::DType) -> Result<Self, Self::Error> {
        match value
            .dtype_type
            .as_ref()
            .ok_or_else(|| vortex_err!(InvalidSerde: "Unrecognized DType"))?
        {
            DtypeType::Null(_) => Ok(Self::Null),
            DtypeType::Bool(b) => Ok(Self::Bool(b.nullable.into())),
            DtypeType::Primitive(p) => Ok(Self::Primitive(p.r#type().into(), p.nullable.into())),
            DtypeType::Decimal(d) => Ok(Self::Decimal(
                DecimalDType::try_new(
                    d.precision.try_into().map_err(|_| vortex_err!("proto precision could not be downcast to u8"))?,
                    d.scale.try_into().map_err(|_| vortex_err!("proto scale could not be downcast to i8"))?)?,
                d.nullable.into())),
            DtypeType::Utf8(u) => Ok(Self::Utf8(u.nullable.into())),
            DtypeType::Binary(b) => Ok(Self::Binary(b.nullable.into())),
            DtypeType::Struct(s) => Ok(Self::Struct(
                StructFields::new(
                    s.names.iter().map(|s| s.as_str()).collect(),
                    s.dtypes
                        .iter()
                        .map(TryInto::<Self>::try_into)
                        .collect::<VortexResult<Vec<_>>>()?,
                ),
                s.nullable.into(),
            )),
            DtypeType::List(l) => {
                let nullable = l.nullable.into();
                Ok(Self::List(
                    l.element_type
                        .as_ref()
                        .ok_or_else(|| vortex_err!(InvalidSerde: "Invalid list element type"))?
                        .as_ref()
                        .try_into()
                        .map(Arc::new)?,
                    nullable,
                ))
            }
            DtypeType::FixedSizeList(fsl) => {
                let nullable = fsl.nullable.into();
                Ok(Self::FixedSizeList(
                    fsl.element_type
                        .as_ref()
                        .ok_or_else(|| vortex_err!(InvalidSerde: "Invalid fixed-size list element type"))?
                        .as_ref()
                        .try_into()
                        .map(Arc::new)?,
                    fsl.size,
                    nullable,
                ))
            }
            DtypeType::Extension(e) => Ok(Self::Extension(
                Arc::new(ExtDType::new(
                    ExtID::from(e.id.as_str()),
                    Arc::new(DType::try_from(e.storage_dtype
                                                 .as_ref()
                                                 .ok_or_else(|| vortex_err!(InvalidSerde: "storage_dtype must be provided in DType proto message"))?
                                                 .as_ref(),
                    ).map_err(|e| vortex_err!("failed converting DType from proto message: {}", e))?),
                    e.metadata.as_ref().map(|m| ExtMetadata::from(m.as_ref())),
                ),
            ))),
        }
    }
}

impl From<&DType> for pb::DType {
    fn from(value: &DType) -> Self {
        Self {
            dtype_type: Some(match value {
                DType::Null => DtypeType::Null(pb::Null {}),
                DType::Bool(null) => DtypeType::Bool(pb::Bool {
                    nullable: (*null).into(),
                }),
                DType::Primitive(ptype, null) => DtypeType::Primitive(pb::Primitive {
                    r#type: pb::PType::from(*ptype).into(),
                    nullable: (*null).into(),
                }),
                DType::Decimal(decimal, null) => DtypeType::Decimal(pb::Decimal {
                    precision: decimal.precision() as u32,
                    scale: decimal.scale() as i32,
                    nullable: (*null).into(),
                }),
                DType::Utf8(null) => DtypeType::Utf8(pb::Utf8 {
                    nullable: (*null).into(),
                }),
                DType::Binary(null) => DtypeType::Binary(pb::Binary {
                    nullable: (*null).into(),
                }),
                DType::Struct(s, null) => DtypeType::Struct(pb::Struct {
                    names: s.names().iter().map(|s| s.as_ref().to_string()).collect(),
                    dtypes: s.fields().map(|d| Self::from(&d)).collect(),
                    nullable: (*null).into(),
                }),
                DType::List(edt, null) => DtypeType::List(Box::new(pb::List {
                    element_type: Some(Box::new(edt.as_ref().into())),
                    nullable: (*null).into(),
                })),
                DType::FixedSizeList(edt, size, null) => {
                    DtypeType::FixedSizeList(Box::new(pb::FixedSizeList {
                        element_type: Some(Box::new(edt.as_ref().into())),
                        size: *size,
                        nullable: (*null).into(),
                    }))
                }
                DType::Extension(e) => DtypeType::Extension(Box::new(pb::Extension {
                    id: e.id().as_ref().into(),
                    storage_dtype: Some(Box::new(e.storage_dtype().into())),
                    metadata: e.metadata().map(|m| m.as_ref().into()),
                })),
            }),
        }
    }
}

impl From<pb::PType> for PType {
    fn from(value: pb::PType) -> Self {
        use pb::PType::*;
        match value {
            U8 => Self::U8,
            U16 => Self::U16,
            U32 => Self::U32,
            U64 => Self::U64,
            I8 => Self::I8,
            I16 => Self::I16,
            I32 => Self::I32,
            I64 => Self::I64,
            F16 => Self::F16,
            F32 => Self::F32,
            F64 => Self::F64,
        }
    }
}

impl From<PType> for pb::PType {
    fn from(value: PType) -> Self {
        use pb::PType::*;
        match value {
            PType::U8 => U8,
            PType::U16 => U16,
            PType::U32 => U32,
            PType::U64 => U64,
            PType::I8 => I8,
            PType::I16 => I16,
            PType::I32 => I32,
            PType::I64 => I64,
            PType::F16 => F16,
            PType::F32 => F32,
            PType::F64 => F64,
        }
    }
}

impl TryFrom<&pb::FieldPath> for FieldPath {
    type Error = VortexError;

    fn try_from(value: &pb::FieldPath) -> Result<Self, Self::Error> {
        let mut path = Vec::with_capacity(value.path.len());
        for field in value.path.iter() {
            match field
                .field_type
                .as_ref()
                .ok_or_else(|| vortex_err!(InvalidSerde: "FieldPath part missing type"))?
            {
                FieldType::Name(name) => path.push(Field::from(name.as_str())),
            }
        }
        Ok(FieldPath::from(path))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::proto::dtype::d_type::DtypeType;
    use crate::proto::dtype::field::FieldType;
    use crate::{
        DType, DecimalDType, ExtDType, ExtID, ExtMetadata, Field, FieldPath, Nullability, PType,
        StructFields,
    };

    fn round_trip_dtype(dtype: &DType) -> DType {
        let pb_dtype = pb::DType::from(dtype);
        DType::try_from(&pb_dtype).expect("Failed to convert from protobuf")
    }

    #[test]
    fn test_primitive_types_round_trip() {
        let test_cases = vec![
            DType::Null,
            DType::Bool(Nullability::NonNullable),
            DType::Bool(Nullability::Nullable),
            DType::Primitive(PType::U8, Nullability::NonNullable),
            DType::Primitive(PType::I32, Nullability::Nullable),
            DType::Primitive(PType::F64, Nullability::NonNullable),
            DType::Utf8(Nullability::Nullable),
            DType::Binary(Nullability::NonNullable),
        ];

        for dtype in test_cases {
            let converted = round_trip_dtype(&dtype);
            assert_eq!(dtype, converted, "Failed for dtype: {:?}", dtype);
        }
    }

    #[test]
    fn test_decimal_round_trip() {
        let decimal_types = vec![
            DType::Decimal(DecimalDType::new(10, 2), Nullability::NonNullable),
            DType::Decimal(DecimalDType::new(38, -5), Nullability::Nullable),
            DType::Decimal(DecimalDType::new(76, 20), Nullability::NonNullable),
        ];

        for dtype in decimal_types {
            let converted = round_trip_dtype(&dtype);
            assert_eq!(dtype, converted);
        }
    }

    #[test]
    fn test_struct_round_trip() {
        let struct_dtype = DType::Struct(
            StructFields::from_iter([
                ("id", DType::Primitive(PType::I64, Nullability::NonNullable)),
                ("name", DType::Utf8(Nullability::Nullable)),
                ("active", DType::Bool(Nullability::NonNullable)),
            ]),
            Nullability::NonNullable,
        );

        let converted = round_trip_dtype(&struct_dtype);
        assert_eq!(struct_dtype, converted);
    }

    #[test]
    fn test_nested_struct_round_trip() {
        let inner_struct = DType::Struct(
            StructFields::from_iter([
                ("street", DType::Utf8(Nullability::NonNullable)),
                ("city", DType::Utf8(Nullability::NonNullable)),
            ]),
            Nullability::Nullable,
        );

        let outer_struct = DType::Struct(
            StructFields::from_iter([
                ("name", DType::Utf8(Nullability::NonNullable)),
                ("address", inner_struct),
            ]),
            Nullability::NonNullable,
        );

        let converted = round_trip_dtype(&outer_struct);
        assert_eq!(outer_struct, converted);
    }

    #[test]
    fn test_list_round_trip() {
        let list_types = vec![
            // List types
            DType::List(
                Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
                Nullability::Nullable,
            ),
            DType::List(
                Arc::new(DType::Utf8(Nullability::Nullable)),
                Nullability::NonNullable,
            ),
            DType::List(
                Arc::new(DType::List(
                    Arc::new(DType::Bool(Nullability::NonNullable)),
                    Nullability::Nullable,
                )),
                Nullability::NonNullable,
            ),
            // FixedSizeList types
            DType::FixedSizeList(
                Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
                3,
                Nullability::Nullable,
            ),
            DType::FixedSizeList(
                Arc::new(DType::Utf8(Nullability::Nullable)),
                5,
                Nullability::NonNullable,
            ),
            DType::FixedSizeList(
                Arc::new(DType::FixedSizeList(
                    Arc::new(DType::Primitive(PType::F64, Nullability::NonNullable)),
                    2,
                    Nullability::Nullable,
                )),
                4,
                Nullability::NonNullable,
            ),
        ];

        for dtype in list_types {
            let converted = round_trip_dtype(&dtype);
            assert_eq!(dtype, converted);
        }
    }

    #[test]
    fn test_extension_round_trip() {
        let ext_dtype = DType::Extension(Arc::new(ExtDType::new(
            ExtID::from("test.extension"),
            Arc::new(DType::Binary(Nullability::NonNullable)),
            Some(ExtMetadata::from(b"test metadata".as_slice())),
        )));

        let converted = round_trip_dtype(&ext_dtype);
        assert_eq!(ext_dtype, converted);
    }

    #[test]
    fn test_field_path_round_trip() {
        let test_paths = vec![
            FieldPath::root(),
            FieldPath::from(vec![Field::from("field1")]),
            FieldPath::from(vec![
                Field::from("field1"),
                Field::from("field2"),
                Field::from("field3"),
            ]),
        ];

        for path in test_paths {
            let pb_path = pb::FieldPath {
                path: path
                    .parts()
                    .iter()
                    .map(|f| pb::Field {
                        field_type: Some(FieldType::Name(f.as_name().unwrap().to_string())),
                    })
                    .collect(),
            };

            let converted = FieldPath::try_from(&pb_path).expect("Failed to convert FieldPath");
            assert_eq!(path, converted);
        }
    }

    #[test]
    fn test_ptype_conversions() {
        let ptypes = vec![
            PType::U8,
            PType::U16,
            PType::U32,
            PType::U64,
            PType::I8,
            PType::I16,
            PType::I32,
            PType::I64,
            PType::F16,
            PType::F32,
            PType::F64,
        ];

        for ptype in ptypes {
            let pb_ptype = pb::PType::from(ptype);
            let converted = PType::from(pb_ptype);
            assert_eq!(ptype, converted);
        }
    }

    #[test]
    fn test_invalid_decimal_from_proto() {
        // Test precision that doesn't fit in u8
        let pb_dtype = pb::DType {
            dtype_type: Some(DtypeType::Decimal(pb::Decimal {
                precision: 300, // Too large for u8
                scale: 2,
                nullable: false,
            })),
        };

        let result = DType::try_from(&pb_dtype);
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_dtype_type() {
        let pb_dtype = pb::DType { dtype_type: None };

        let result = DType::try_from(&pb_dtype);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Unrecognized DType")
        );
    }

    #[test]
    fn test_missing_list_element() {
        let pb_dtype = pb::DType {
            dtype_type: Some(DtypeType::List(Box::new(pb::List {
                element_type: None,
                nullable: false,
            }))),
        };

        let result = DType::try_from(&pb_dtype);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Invalid list element type")
        );
    }

    #[test]
    fn test_missing_extension_storage() {
        let pb_dtype = pb::DType {
            dtype_type: Some(DtypeType::Extension(Box::new(pb::Extension {
                id: "test.ext".to_string(),
                storage_dtype: None,
                metadata: None,
            }))),
        };

        let result = DType::try_from(&pb_dtype);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("storage_dtype must be provided")
        );
    }
}
