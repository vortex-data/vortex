SELECT p_partkey, p_name, p_retailprice 
FROM part 
WHERE p_size > 10 
ORDER BY p_retailprice DESC 
LIMIT 10;
