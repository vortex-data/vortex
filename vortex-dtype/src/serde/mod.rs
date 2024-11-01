#[cfg(feature = "flatbuffers")]
pub mod flatbuffers;
#[cfg(feature = "proto")]
mod proto;
#[allow(clippy::module_inception)]
mod serde;

#[cfg(test)]
#[cfg(feature = "serde")]
mod test {
    use serde_test::{assert_tokens, Token};

    use crate::{DType, Nullability, PType};

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
        assert_tokens(&DType::from(PType::U8), &[Token::Struct {
            name: "DType",
            len: 2,
        }]);
    }

    #[test]
    fn test_serde_nullability() {
        assert_tokens(&Nullability::NonNullable, &[Token::UnitVariant {
            name: "Nullability",
            variant: "non_nullable",
        }]);
    }
}
