
SELECT 
    a.name AS actor_name, 
    t.title AS movie_title, 
    c.kind AS cast_type, 
    a.id AS actor_id, 
    t.production_year, 
    STRING_AGG(k.keyword, ', ') AS keywords 
FROM 
    aka_name a 
JOIN 
    cast_info ci ON a.person_id = ci.person_id 
JOIN 
    title t ON ci.movie_id = t.id 
JOIN 
    comp_cast_type c ON ci.person_role_id = c.id 
LEFT JOIN 
    movie_keyword mk ON t.id = mk.movie_id 
LEFT JOIN 
    keyword k ON mk.keyword_id = k.id 
WHERE 
    t.production_year >= 2000 
    AND a.name IS NOT NULL 
GROUP BY 
    a.name, t.title, c.kind, a.id, t.production_year 
ORDER BY 
    t.production_year DESC, a.name;
