// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serde serialization and deserialization for DTypes

pub(crate) mod flatbuffers;

mod proto;

#[expect(clippy::module_inception)]
#[cfg(feature = "serde")]
mod serde;

#[cfg(feature = "serde")]
pub use serde::*;

#[cfg(test)]
#[cfg(feature = "serde")]
mod test {
    use serde::de::DeserializeSeed;
    use serde_test::Token;
    use serde_test::assert_tokens;

    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::serde::DTypeSerde;
    use crate::dtype::test::SESSION;

    #[test]
    fn test_serde_ptype_json() {
        // Ensure we serialize PTypes to lowercase.
        let serialized = serde_json::to_string(&PType::U8).unwrap();
        assert_eq!(serialized, "\"u8\"");
        assert_eq!(serde_json::from_str::<PType>("\"u8\"").unwrap(), PType::U8);
    }

    #[test]
    fn test_serde_ptype() {
        assert_tokens(
            &PType::U8,
            &[Token::UnitVariant {
                name: "PType",
                variant: "u8",
            }],
        );
    }

    #[test]
    fn test_serde_dtype() {
        // Test that DType serializes correctly (we only test serialization since
        // deserialization with serde_test has borrowing issues with enum variants)
        use serde_test::assert_ser_tokens;
        assert_ser_tokens(
            &DType::from(PType::U8),
            &[
                Token::TupleVariant {
                    name: "DType",
                    variant: "Primitive",
                    len: 2,
                },
                Token::UnitVariant {
                    name: "PType",
                    variant: "u8",
                },
                Token::Bool(false),
                Token::TupleVariantEnd,
            ],
        );
    }

    #[test]
    fn test_serde_variant_dtype() {
        use serde_test::assert_ser_tokens;

        assert_ser_tokens(
            &DType::Variant(Nullability::NonNullable),
            &[
                Token::NewtypeVariant {
                    name: "DType",
                    variant: "Variant",
                },
                Token::Bool(false),
            ],
        );
    }

    #[test]
    fn test_serde_nullability() {
        assert_tokens(&Nullability::NonNullable, &[Token::Bool(false)]);
    }

    #[test]
    fn test_serde_struct_dtype_json() {
        use crate::dtype::StructFields;

        // Create a struct dtype with various field types
        let fields = StructFields::from_iter([
            ("name", DType::Utf8(Nullability::NonNullable)),
            ("age", DType::Primitive(PType::I32, Nullability::Nullable)),
            ("active", DType::Bool(Nullability::NonNullable)),
        ]);
        let struct_dtype = DType::Struct(fields, Nullability::Nullable);

        // Serialize to JSON
        let json = serde_json::to_string_pretty(&struct_dtype).unwrap();

        // Assert the JSON representation hasn't changed
        insta::assert_snapshot!(json, @r#"
        {
          "Struct": [
            {
              "names": [
                "name",
                "age",
                "active"
              ],
              "dtypes": [
                {
                  "Utf8": false
                },
                {
                  "Primitive": [
                    "i32",
                    true
                  ]
                },
                {
                  "Bool": false
                }
              ]
            },
            true
          ]
        }
        "#);

        // Deserialize back and verify round-trip
        let mut deserializer = serde_json::Deserializer::from_str(&json);
        let deserialized: DType = DTypeSerde::<DType>::new(&SESSION)
            .deserialize(&mut deserializer)
            .unwrap();
        assert_eq!(struct_dtype, deserialized);
    }

    #[test]
    fn test_serde_struct_fields_from_json_value() {
        use serde::de::IntoDeserializer;

        use crate::dtype::StructFields;

        let fields = StructFields::from_iter([
            ("name", DType::Utf8(Nullability::NonNullable)),
            ("age", DType::Primitive(PType::I32, Nullability::Nullable)),
        ]);

        let value: serde_json::Value = serde_json::to_value(&fields).unwrap();

        let json_str = value.to_string();
        let mut deserializer = serde_json::Deserializer::from_str(&json_str);
        let from_str: StructFields = DTypeSerde::<StructFields>::new(&SESSION)
            .deserialize(&mut deserializer)
            .unwrap();
        assert_eq!(fields, from_str);

        let deserializer = value.into_deserializer();
        let from_value: StructFields = DTypeSerde::<StructFields>::new(&SESSION)
            .deserialize(deserializer)
            .unwrap();
        assert_eq!(fields, from_value);
    }

    #[test]
    fn test_serde_variant_dtype_json_roundtrip() {
        let json = serde_json::to_string(&DType::Variant(Nullability::Nullable)).unwrap();
        assert_eq!(json, "{\"Variant\":true}");

        let mut deserializer = serde_json::Deserializer::from_str(&json);
        let deserialized: DType = DTypeSerde::<DType>::new(&SESSION)
            .deserialize(&mut deserializer)
            .unwrap();
        assert_eq!(DType::Variant(Nullability::Nullable), deserialized);
    }
}
