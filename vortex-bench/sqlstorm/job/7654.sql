
SELECT 
    a.name AS actor_name,
    t.title AS movie_title,
    c1.kind AS company_type,
    c2.name AS company_name,
    t.production_year,
    COUNT(DISTINCT k.keyword) AS keyword_count
FROM 
    aka_name a
JOIN 
    cast_info ci ON a.person_id = ci.person_id
JOIN 
    title t ON ci.movie_id = t.id
JOIN 
    movie_companies mc ON t.id = mc.movie_id
JOIN 
    company_type c1 ON mc.company_type_id = c1.id
JOIN 
    company_name c2 ON mc.company_id = c2.id
LEFT JOIN 
    movie_keyword mk ON t.id = mk.movie_id
LEFT JOIN 
    keyword k ON mk.keyword_id = k.id
WHERE 
    t.production_year BETWEEN 2000 AND 2020
    AND c2.country_code = 'USA'
GROUP BY 
    a.name, t.title, c1.kind, c2.name, t.production_year
ORDER BY 
    keyword_count DESC, actor_name ASC;
