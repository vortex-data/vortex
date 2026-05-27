WITH RankedPosts AS (
    SELECT 
        p.Id AS PostId,
        p.Title,
        p.CreationDate,
        p.Score,
        p.ViewCount,
        ROW_NUMBER() OVER (PARTITION BY p.PostTypeId ORDER BY p.Score DESC) AS PostRank
    FROM 
        Posts p
    WHERE 
        p.CreationDate >= cast('2024-10-01 12:34:56' as timestamp) - INTERVAL '1 year'
),
PostVotes AS (
    SELECT 
        v.PostId,
        COUNT(CASE WHEN v.VoteTypeId = 2 THEN 1 END) AS UpVotesCount,
        COUNT(CASE WHEN v.VoteTypeId = 3 THEN 1 END) AS DownVotesCount
    FROM 
        Votes v
    GROUP BY 
        v.PostId
),
FilteredPosts AS (
    SELECT 
        r.PostId,
        r.Title,
        r.CreationDate,
        r.Score,
        r.ViewCount,
        COALESCE(pv.UpVotesCount, 0) AS UpVotes,
        COALESCE(pv.DownVotesCount, 0) AS DownVotes
    FROM 
        RankedPosts r
    LEFT JOIN 
        PostVotes pv ON r.PostId = pv.PostId
    WHERE 
        r.PostRank <= 10
),
CommentsInfo AS (
    SELECT 
        c.PostId,
        COUNT(c.Id) AS CommentCount,
        STRING_AGG(c.Text, '; ') AS CommentTexts
    FROM 
        Comments c
    GROUP BY 
        c.PostId
)
SELECT 
    fp.PostId,
    fp.Title,
    fp.CreationDate,
    fp.Score,
    fp.ViewCount,
    fp.UpVotes,
    fp.DownVotes,
    COALESCE(ci.CommentCount, 0) AS TotalComments,
    COALESCE(ci.CommentTexts, '') AS LastCommentsSnippet
FROM 
    FilteredPosts fp
LEFT JOIN 
    CommentsInfo ci ON fp.PostId = ci.PostId
ORDER BY 
    fp.Score DESC, fp.ViewCount ASC;