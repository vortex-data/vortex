SELECT 
    akn.name AS aka_name, 
    tit.title AS movie_title, 
    cnt.name AS company_name, 
    rt.role AS person_role, 
    pi.info AS person_info
FROM 
    aka_name akn 
JOIN 
    cast_info ci ON akn.person_id = ci.person_id 
JOIN 
    title tit ON ci.movie_id = tit.id 
JOIN 
    movie_companies mc ON tit.id = mc.movie_id 
JOIN 
    company_name cnt ON mc.company_id = cnt.id 
JOIN 
    role_type rt ON ci.role_id = rt.id 
JOIN 
    person_info pi ON akn.person_id = pi.person_id 
WHERE 
    tit.production_year >= 2000 
    AND cnt.country_code = 'USA' 
    AND pi.info_type_id IN (SELECT id FROM info_type WHERE info = 'Biography') 
ORDER BY 
    tit.production_year DESC, akn.name;
