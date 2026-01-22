// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod flatbuffers;

// mod proto;

#[allow(clippy::module_inception)]
#[cfg(feature = "serde")]
mod serde;

#[cfg(test)]
#[cfg(feature = "serde")]
mod test {
    use serde_test::Token;
    use serde_test::assert_tokens;

    use crate::DType;
    use crate::Nullability;
    use crate::PType;

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
        assert_tokens(
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
    fn test_serde_nullability() {
        assert_tokens(&Nullability::NonNullable, &[Token::Bool(false)]);
    }

    #[test]
    fn test_serde_struct_dtype_json() {
        use crate::StructFields;

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
                  "inner": {
                    "Owned": {
                      "Utf8": false
                    }
                  }
                },
                {
                  "inner": {
                    "Owned": {
                      "Primitive": [
                        "i32",
                        true
                      ]
                    }
                  }
                },
                {
                  "inner": {
                    "Owned": {
                      "Bool": false
                    }
                  }
                }
              ]
            },
            true
          ]
        }
        "#);

        // Deserialize back and verify round-trip
        let deserialized: DType = serde_json::from_str(&json).unwrap();
        assert_eq!(struct_dtype, deserialized);
    }
}
