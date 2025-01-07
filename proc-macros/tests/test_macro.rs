#![allow(unused)]
// TODO(aduffy): add tests for the macro

use proc_macro_traits::{AvroValue, FromAvro};
use proc_macros::FromAvro;

#[test]
fn test_derive_macro() {
    #[derive(FromAvro)]
    struct MyRecordType {
        a: i32,
        b: String,
    }

    let value = AvroValue::Record(vec![
        ("a".to_string(), AvroValue::Int(1)),
        ("b".to_string(), AvroValue::String("hello".to_string())),
    ]);

    let record: MyRecordType = MyRecordType::try_from(value).unwrap();
    assert_eq!(record.a, 1);
    assert_eq!(record.b, "hello");
}
