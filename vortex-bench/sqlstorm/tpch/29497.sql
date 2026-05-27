WITH Supplier_Products AS (
    SELECT 
        s.s_name AS supplier_name,
        p.p_name AS product_name,
        p.p_brand AS product_brand,
        p.p_container AS product_container,
        ps.ps_availqty AS available_quantity,
        ps.ps_supplycost AS supply_cost,
        p.p_comment AS product_comment
    FROM 
        supplier s
    JOIN 
        partsupp ps ON s.s_suppkey = ps.ps_suppkey
    JOIN 
        part p ON ps.ps_partkey = p.p_partkey
),
Aggregated_Supplier_Products AS (
    SELECT 
        supplier_name,
        COUNT(*) AS total_products,
        SUM(available_quantity) AS total_available_quantity,
        AVG(supply_cost) AS average_supply_cost,
        STRING_AGG(DISTINCT product_brand, ', ') AS brands_offered,
        STRING_AGG(DISTINCT product_container, ', ') AS container_types
    FROM 
        Supplier_Products
    GROUP BY 
        supplier_name
)
SELECT 
    supplier_name,
    total_products,
    total_available_quantity,
    average_supply_cost,
    brands_offered,
    container_types,
    CONCAT('Supplier: ', supplier_name, ' offers ', total_products, ' products, with an average supply cost of $', ROUND(average_supply_cost, 2), '.') AS supplier_summary
FROM 
    Aggregated_Supplier_Products
WHERE 
    total_available_quantity > 0
ORDER BY 
    total_products DESC;
