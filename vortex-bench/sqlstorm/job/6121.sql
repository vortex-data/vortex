
SELECT 
    a.name AS actor_name, 
    t.title AS movie_title, 
    t.production_year, 
    STRING_AGG(DISTINCT k.keyword, ',' ORDER BY k.keyword) AS keywords, 
    c.kind AS company_type, 
    ci.role_id 
FROM 
    aka_name a 
JOIN 
    cast_info ci ON a.person_id = ci.person_id 
JOIN 
    aka_title t ON ci.movie_id = t.movie_id 
JOIN 
    movie_keyword mk ON mk.movie_id = t.id 
JOIN 
    keyword k ON mk.keyword_id = k.id 
JOIN 
    movie_companies mc ON t.id = mc.movie_id 
JOIN 
    company_type c ON mc.company_type_id = c.id 
WHERE 
    t.production_year >= 2000 
AND 
    c.kind LIKE 'Production%' 
GROUP BY 
    a.name, 
    t.title, 
    t.production_year, 
    c.kind, 
    ci.role_id 
ORDER BY 
    t.production_year DESC, 
    a.name;
