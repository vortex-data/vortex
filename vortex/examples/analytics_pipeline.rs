//! Analytics Pipeline Example
//!
//! This example demonstrates building a complete analytics pipeline with Vortex,
//! including data generation, transformation, aggregation, and persistence.
//!
//! Use case: E-commerce transaction analytics
//!
//! Run with: cargo run --example analytics_pipeline --features tokio

use std::collections::HashMap;

use vortex::arrays::{PrimitiveArray, StructArray, VarBinArray};
use vortex::compressor::CompactCompressor;
use vortex::compute::{filter, sum};
use vortex::dtype::{DType, Nullability};
use vortex::file::{VortexOpenOptions, VortexWriteOptions, WriteStrategyBuilder};
use vortex::stream::ArrayStreamExt;
use vortex::validity::Validity;
use vortex::{Array, ArrayLen, IntoArray, IntoCanonical};
use vortex_expr::operators::gt;
use vortex_expr::root;
use vortex_scalar::Scalar;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== E-Commerce Analytics Pipeline ===\n");
    println!("This example simulates an analytics pipeline for e-commerce transactions.\n");

    // Step 1: Generate synthetic transaction data
    println!("Step 1: Generating transaction data...");
    let transactions = generate_transactions(1000);
    println!("   Generated {} transactions", transactions.len());

    // Step 2: Write data to a Vortex file with compression
    println!("\nStep 2: Persisting data to disk...");
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("ecommerce_transactions.vortex");
    write_transactions(&file_path, &transactions).await?;

    let file_size = std::fs::metadata(&file_path)?.len();
    println!(
        "   File written: {} bytes ({:.2} KB)",
        file_size,
        file_size as f64 / 1024.0
    );

    // Step 3: Query and analyze the data
    println!("\nStep 3: Running analytics queries...");

    println!("\n  Query 1: Total revenue");
    calculate_total_revenue(&transactions)?;

    println!("\n  Query 2: Revenue by category");
    revenue_by_category(&transactions)?;

    println!("\n  Query 3: High-value transactions (amount > $500)");
    high_value_transactions(&transactions)?;

    // Step 4: Filter and export specific data
    println!("\nStep 4: Reading filtered data from disk...");
    filter_from_file(&file_path).await?;

    // Clean up
    std::fs::remove_file(&file_path)?;

    println!("\n=== Analytics pipeline completed successfully! ===");
    Ok(())
}

fn generate_transactions(count: usize) -> StructArray {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let categories = vec!["Electronics", "Clothing", "Books", "Home", "Sports"];
    let customers = vec!["CUST001", "CUST002", "CUST003", "CUST004", "CUST005"];

    let mut category_list = Vec::with_capacity(count);
    let mut customer_list = Vec::with_capacity(count);
    let mut amounts = Vec::with_capacity(count);
    let mut quantities = Vec::with_capacity(count);

    for _ in 0..count {
        customer_list.push(customers[rng.gen_range(0..customers.len())]);
        category_list.push(categories[rng.gen_range(0..categories.len())]);
        amounts.push(rng.gen_range(10.0..1000.0));
        quantities.push(rng.gen_range(1..10) as u32);
    }

    // Create arrays
    let customer_array =
        VarBinArray::from_iter(customer_list.iter(), DType::Utf8(Nullability::NonNullable));
    let category_array =
        VarBinArray::from_iter(category_list.iter(), DType::Utf8(Nullability::NonNullable));
    let amount_array: PrimitiveArray = PrimitiveArray::from(amounts);
    let quantity_array: PrimitiveArray = PrimitiveArray::from(quantities);

    StructArray::try_new(
        ["customer_id", "category", "amount", "quantity"].into(),
        vec![
            customer_array.into_array(),
            category_array.into_array(),
            amount_array.into_array(),
            quantity_array.into_array(),
        ],
        count,
        Validity::NonNullable,
    )
    .expect("Failed to create struct array")
}

async fn write_transactions(
    path: impl AsRef<std::path::Path>,
    data: &StructArray,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = tokio::fs::File::create(path).await?;

    let write_opts = VortexWriteOptions::default().with_strategy(
        WriteStrategyBuilder::new()
            .with_compressor(CompactCompressor::default())
            .build(),
    );

    write_opts
        .write(&mut file, data.clone().to_array_stream())
        .await?;

    Ok(())
}

fn calculate_total_revenue(transactions: &StructArray) -> Result<(), Box<dyn std::error::Error>> {
    let amount_field = transactions.field_by_name("amount")?;

    // Calculate sum
    let total = sum(&amount_field)?;

    if let Scalar::Primitive(prim) = total.as_ref() {
        if let Ok(value) = prim.typed_value::<f64>() {
            println!("   Total revenue: ${:.2}", value);
        }
    }

    Ok(())
}

fn revenue_by_category(transactions: &StructArray) -> Result<(), Box<dyn std::error::Error>> {
    let category_field = transactions.field_by_name("category")?;
    let amount_field = transactions.field_by_name("amount")?;

    // Extract categories and amounts via canonical form
    let categories_canonical = category_field
        .clone()
        .into_canonical()?
        .into_varbin()
        .ok_or("Expected varbin")?;
    let amounts_canonical = amount_field
        .clone()
        .into_canonical()?
        .into_primitive()
        .ok_or("Expected primitive")?;

    // Group by category and sum
    let mut category_revenue: HashMap<String, f64> = HashMap::new();

    for i in 0..categories_canonical.len() {
        let category_bytes = categories_canonical.bytes_at(i).ok_or("Invalid category")?;
        let category = String::from_utf8(category_bytes.to_vec())?;
        let amount = amounts_canonical.get_as::<f64>(i).ok_or("Invalid amount")?;

        *category_revenue.entry(category).or_insert(0.0) += amount;
    }

    println!("   Revenue by category:");
    for (category, revenue) in category_revenue.iter() {
        println!("     {}: ${:.2}", category, revenue);
    }

    Ok(())
}

fn high_value_transactions(transactions: &StructArray) -> Result<(), Box<dyn std::error::Error>> {
    // Filter for amount > 500
    let filter_expr = gt(root().field("amount"), 500.0f64);

    // Apply filter
    let filtered = filter(transactions.as_ref(), &filter_expr)?;

    println!(
        "   High-value transactions (> $500): {} found",
        filtered.len()
    );

    Ok(())
}

async fn filter_from_file(
    path: impl AsRef<std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Read transactions with filter for high-value items
    let reader = VortexOpenOptions::new().open(path).await?;

    let filter = gt(root().field("amount"), 500.0f64);

    let scan = reader.scan()?.with_filter(filter);
    let array = scan.into_array_stream()?.read_all().await?;

    println!("   Read {} high-value transactions from disk", array.len());
    println!("   (Filter applied during read for efficiency)");

    Ok(())
}
