
WITH AddressInfo AS (
    SELECT 
        ca_city,
        ca_state,
        COUNT(*) AS address_count,
        STRING_AGG(DISTINCT CONCAT(ca_street_number, ' ', ca_street_name, ' ', ca_street_type), ', ') AS street_info
    FROM customer_address
    GROUP BY ca_city, ca_state
),
CustomerGender AS (
    SELECT 
        cd_gender,
        COUNT(*) AS gender_count
    FROM customer_demographics
    GROUP BY cd_gender
),
DateStats AS (
    SELECT 
        d_year,
        COUNT(*) AS sales_count,
        SUM(EXTRACT(DOY FROM d_date)) AS total_days_of_year
    FROM date_dim
    JOIN web_sales ON d_date_sk = ws_sold_date_sk
    GROUP BY d_year
),
WarehouseInfo AS (
    SELECT 
        w_city,
        AVG(w_warehouse_sq_ft) AS avg_warehouse_size
    FROM warehouse
    GROUP BY w_city
),
FinalBenchmark AS (
    SELECT 
        ai.ca_city,
        ai.ca_state,
        ai.address_count,
        ai.street_info,
        cg.cd_gender,
        cg.gender_count,
        ds.d_year,
        ds.sales_count,
        ds.total_days_of_year,
        wi.w_city,
        wi.avg_warehouse_size
    FROM AddressInfo ai
    JOIN CustomerGender cg ON TRUE
    JOIN DateStats ds ON TRUE
    JOIN WarehouseInfo wi ON wi.w_city = ai.ca_city
)
SELECT 
    CONCAT(ca_city, ', ', ca_state) AS location,
    address_count,
    street_info,
    gender_count,
    d_year,
    sales_count,
    ROUND(total_days_of_year::decimal / NULLIF(sales_count, 0), 2) AS avg_sales_per_day,
    ROUND(avg_warehouse_size::decimal, 2) AS average_warehouse_size
FROM FinalBenchmark
ORDER BY location, d_year;
