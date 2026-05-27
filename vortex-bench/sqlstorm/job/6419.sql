
SELECT 
    a.name AS actor_name,
    t.title AS movie_title,
    t.production_year,
    STRING_AGG(DISTINCT g.keyword, ', ' ORDER BY g.keyword) AS keywords,
    c.kind AS company_type,
    r.role AS role_in_movie,
    pi.info AS person_info
FROM 
    aka_name a
JOIN 
    cast_info ci ON a.person_id = ci.person_id
JOIN 
    title t ON ci.movie_id = t.id
JOIN 
    movie_keyword mk ON t.id = mk.movie_id
JOIN 
    keyword g ON mk.keyword_id = g.id
JOIN 
    movie_companies mc ON t.id = mc.movie_id
JOIN 
    company_name cn ON mc.company_id = cn.id
JOIN 
    company_type c ON mc.company_type_id = c.id
JOIN 
    role_type r ON ci.role_id = r.id
LEFT JOIN 
    person_info pi ON a.person_id = pi.person_id 
WHERE 
    t.production_year >= 2000 
    AND c.kind LIKE 'Distributor%'
GROUP BY 
    a.name, t.title, t.production_year, c.kind, r.role, pi.info
ORDER BY 
    t.production_year DESC, a.name;
