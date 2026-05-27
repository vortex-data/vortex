
WITH CustomerSales AS (
    SELECT
        c.c_customer_sk,
        c.c_first_name,
        c.c_last_name,
        SUM(ws.ws_net_profit) AS total_profit,
        RANK() OVER (PARTITION BY ca.ca_state ORDER BY SUM(ws.ws_net_profit) DESC) AS rank_within_state
    FROM
        customer c
    JOIN
        web_sales ws ON c.c_customer_sk = ws.ws_bill_customer_sk
    JOIN
        customer_address ca ON c.c_current_addr_sk = ca.ca_address_sk
    WHERE
        c.c_current_cdemo_sk IS NOT NULL
    GROUP BY
        c.c_customer_sk, c.c_first_name, c.c_last_name, ca.ca_state
),
TopCustomers AS (
    SELECT
        cs.c_customer_sk,
        cs.c_first_name,
        cs.c_last_name,
        cs.total_profit
    FROM
        CustomerSales cs
    WHERE
        cs.rank_within_state <= 5
),
ProductsSold AS (
    SELECT
        ws.ws_item_sk,
        COUNT(ws.ws_order_number) AS total_orders,
        AVG(ws.ws_sales_price) AS average_price
    FROM
        web_sales ws
    WHERE
        ws.ws_sold_date_sk BETWEEN (SELECT MAX(d_date_sk) - 30 FROM date_dim) AND (SELECT MAX(d_date_sk) FROM date_dim)
    GROUP BY
        ws.ws_item_sk
)
SELECT
    tc.c_first_name,
    tc.c_last_name,
    p.total_orders,
    p.average_price,
    CASE 
        WHEN tc.total_profit IS NULL THEN 'No sales'
        ELSE 'Sales recorded'
    END AS sales_status
FROM
    TopCustomers tc
LEFT JOIN
    ProductsSold p ON tc.c_customer_sk = p.ws_item_sk
ORDER BY
    tc.total_profit DESC;
