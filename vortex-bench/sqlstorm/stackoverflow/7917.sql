WITH RankedPosts AS (
    SELECT 
        p.Id AS PostId,
        p.Title,
        p.CreationDate,
        p.Score,
        p.ViewCount,
        p.AnswerCount,
        COALESCE(u.DisplayName, 'Community') AS OwnerDisplayName,
        ROW_NUMBER() OVER (PARTITION BY p.PostTypeId ORDER BY p.Score DESC) AS Rank
    FROM 
        Posts p
    LEFT JOIN 
        Users u ON p.OwnerUserId = u.Id
    WHERE 
        p.CreationDate >= cast('2024-10-01 12:34:56' as timestamp) - INTERVAL '1 year'
),
PostStats AS (
    SELECT 
        PostId,
        Title,
        CreationDate,
        Score,
        ViewCount,
        AnswerCount,
        OwnerDisplayName,
        Rank
    FROM 
        RankedPosts
    WHERE 
        Rank <= 10
),
CommentsInfo AS (
    SELECT
        c.PostId,
        COUNT(c.Id) AS CommentCount,
        MAX(c.CreationDate) AS LastCommentDate
    FROM 
        Comments c
    GROUP BY 
        c.PostId
),
FinalReport AS (
    SELECT 
        ps.PostId,
        ps.Title,
        ps.CreationDate,
        ps.Score,
        ps.ViewCount,
        ps.AnswerCount,
        ps.OwnerDisplayName,
        ci.CommentCount,
        ci.LastCommentDate
    FROM 
        PostStats ps
    LEFT JOIN 
        CommentsInfo ci ON ps.PostId = ci.PostId
)
SELECT 
    Title,
    CreationDate,
    Score,
    ViewCount,
    AnswerCount,
    OwnerDisplayName,
    COALESCE(CommentCount, 0) AS CommentCount,
    LastCommentDate
FROM 
    FinalReport
ORDER BY 
    Score DESC;