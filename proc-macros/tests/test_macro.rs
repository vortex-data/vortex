use proc_macros::FromTuple;

// We can call it something even if there is no trait

#[test]
fn test_derive_macro() {
    #[derive(FromTuple)]
    struct MyThing {
        name: String,
        age: u32,
        social_security_number: u64,
    }

    let thing1 = MyThing::from(("Andrew".to_string(), 29, 1234567890));
    assert_eq!(thing1.name, "Andrew".to_string());
    assert_eq!(thing1.age, 29);
    assert_eq!(thing1.social_security_number, 1234567890);

    // This should fail at compile time
    #[derive(FromTuple, Debug, PartialEq)]
    struct MyUnitStruct();

    assert_eq!(MyUnitStruct::from(()), MyUnitStruct());
}
