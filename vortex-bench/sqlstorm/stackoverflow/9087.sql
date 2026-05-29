
WITH RankedPosts AS (
    SELECT 
        p.Id AS PostId,
        p.Title,
        p.CreationDate,
        p.Score,
        p.ViewCount,
        u.DisplayName AS OwnerDisplayName,
        RANK() OVER (PARTITION BY p.PostTypeId ORDER BY p.Score DESC, p.CreationDate DESC) AS RankScore,
        COUNT(c.Id) AS CommentCount,
        SUM(CASE WHEN v.VoteTypeId = 2 THEN 1 ELSE 0 END) AS UpvoteCount,
        SUM(CASE WHEN v.VoteTypeId = 3 THEN 1 ELSE 0 END) AS DownvoteCount,
        p.PostTypeId  -- Added to the GROUP BY clause
    FROM 
        Posts p
    LEFT JOIN 
        Users u ON p.OwnerUserId = u.Id
    LEFT JOIN 
        Comments c ON p.Id = c.PostId
    LEFT JOIN 
        Votes v ON p.Id = v.PostId
    GROUP BY 
        p.Id, p.Title, p.CreationDate, p.Score, p.ViewCount, u.DisplayName, p.PostTypeId  -- Added p.PostTypeId
), FilteredPosts AS (
    SELECT 
        rp.*,
        pt.Name AS PostTypeName
    FROM 
        RankedPosts rp
    JOIN 
        PostTypes pt ON rp.PostTypeId = pt.Id
    WHERE 
        rp.RankScore <= 5 
)
SELECT 
    fp.PostId,
    fp.Title,
    fp.PostTypeName,
    fp.CreationDate,
    fp.Score,
    fp.ViewCount,
    fp.CommentCount,
    fp.UpvoteCount,
    fp.DownvoteCount,
    fp.OwnerDisplayName
FROM 
    FilteredPosts fp
ORDER BY 
    fp.PostTypeName, fp.RankScore;
