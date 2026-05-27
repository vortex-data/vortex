
SELECT 
    a.name AS actor_name,
    t.title AS movie_title,
    c.kind AS company_type,
    COUNT(DISTINCT mc.company_id) AS company_count,
    STRING_AGG(DISTINCT kw.keyword, ', ') AS keywords,
    MIN(t.production_year) AS first_year,
    MAX(t.production_year) AS last_year
FROM 
    aka_name a
JOIN 
    cast_info ci ON a.person_id = ci.person_id
JOIN 
    title t ON ci.movie_id = t.id
JOIN 
    movie_companies mc ON t.id = mc.movie_id
JOIN 
    company_type c ON mc.company_type_id = c.id
LEFT JOIN 
    movie_keyword mk ON t.id = mk.movie_id
LEFT JOIN 
    keyword kw ON mk.keyword_id = kw.id
WHERE 
    t.production_year BETWEEN 2000 AND 2020
    AND c.kind = 'Distributor'
GROUP BY 
    a.name, t.title, c.kind
ORDER BY 
    first_year DESC;
