WITH RankedPosts AS (
    SELECT 
        p.Id AS PostId,
        p.Title,
        p.Tags,
        p.Body,
        COUNT(c.Id) AS CommentCount,
        SUM(CASE WHEN v.VoteTypeId = 2 THEN 1 ELSE 0 END) AS UpVotes,
        SUM(CASE WHEN v.VoteTypeId = 3 THEN 1 ELSE 0 END) AS DownVotes,
        ROW_NUMBER() OVER (PARTITION BY p.Tags ORDER BY COUNT(c.Id) DESC) AS Rank
    FROM 
        Posts p
    LEFT JOIN 
        Comments c ON p.Id = c.PostId
    LEFT JOIN 
        Votes v ON p.Id = v.PostId
    WHERE 
        p.PostTypeId = 1 
    GROUP BY 
        p.Id, p.Title, p.Tags, p.Body
),
FilteredPosts AS (
    SELECT 
        PostId,
        Title,
        Tags,
        Body,
        CommentCount,
        UpVotes,
        DownVotes
    FROM 
        RankedPosts
    WHERE 
        Rank <= 5 
)
SELECT 
    fp.Tags,
    STRING_AGG(fp.Title, '; ') AS TopQuestions,
    SUM(fp.CommentCount) AS TotalComments,
    SUM(fp.UpVotes) AS TotalUpVotes,
    SUM(fp.DownVotes) AS TotalDownVotes,
    COUNT(fp.PostId) AS TotalPosts
FROM 
    FilteredPosts fp
GROUP BY 
    fp.Tags
ORDER BY 
    TotalUpVotes DESC
LIMIT 10;