
SELECT ca_state, COUNT(*) as customer_count
FROM customer_address
GROUP BY ca_state
ORDER BY customer_count DESC
LIMIT 10;
