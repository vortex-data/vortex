WITH CustomerDemographics AS (
    SELECT cd_demo_sk, 
           cd_gender, 
           cd_marital_status, 
           cd_education_status, 
           cd_purchase_estimate, 
           cd_credit_rating, 
           cd_dep_count, 
           cd_dep_employed_count, 
           cd_dep_college_count
    FROM customer_demographics
    WHERE cd_marital_status = 'M'
), CustomerDetails AS (
    SELECT c.c_customer_sk, 
           c.c_first_name, 
           c.c_last_name, 
           ca.ca_city, 
           ca.ca_state, 
           ca.ca_country, 
           cd.cd_gender, 
           cd.cd_purchase_estimate
    FROM customer c
    JOIN customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
    JOIN CustomerDemographics cd ON c.c_current_cdemo_sk = cd.cd_demo_sk
), DateFiltered AS (
    SELECT d.d_date, 
           d.d_year, 
           COUNT(ws.ws_order_number) AS total_orders,
           SUM(ws.ws_sales_price) AS total_sales
    FROM web_sales ws
    JOIN date_dim d ON ws.ws_sold_date_sk = d.d_date_sk
    WHERE d.d_year = 2001
    GROUP BY d.d_date, d.d_year
), SalesByCustomer AS (
    SELECT cd.c_customer_sk, 
           cd.c_first_name, 
           cd.c_last_name, 
           cd.ca_city, 
           cd.ca_state, 
           SUM(ws.ws_sales_price) AS customer_sales
    FROM CustomerDetails cd
    JOIN web_sales ws ON cd.c_customer_sk = ws.ws_ship_customer_sk
    GROUP BY cd.c_customer_sk, cd.c_first_name, cd.c_last_name, cd.ca_city, cd.ca_state
), FinalReport AS (
    SELECT dbc.d_date AS Sales_Date, 
           dbc.total_orders, 
           dbc.total_sales, 
           sbc.c_customer_sk, 
           sbc.c_first_name, 
           sbc.c_last_name, 
           sbc.ca_city, 
           sbc.ca_state, 
           sbc.customer_sales
    FROM DateFiltered dbc
    JOIN SalesByCustomer sbc ON dbc.d_date = cast('2002-10-01' as date) 
)
SELECT * 
FROM FinalReport
ORDER BY total_sales DESC, customer_sales DESC;