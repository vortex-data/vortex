WITH RecentPosts AS (
    SELECT 
        P.Id AS PostId,
        P.Title,
        P.CreationDate,
        P.Score,
        P.ViewCount,
        P.AnswerCount,
        U.DisplayName AS OwnerDisplayName,
        COALESCE(V.UpVotes, 0) AS UpVotes,
        COALESCE(V.DownVotes, 0) AS DownVotes,
        DENSE_RANK() OVER (PARTITION BY EXTRACT(YEAR FROM P.CreationDate) ORDER BY P.CreationDate DESC) AS YearRank
    FROM 
        Posts P
    JOIN 
        Users U ON P.OwnerUserId = U.Id
    LEFT JOIN (
        SELECT 
            PostId, 
            SUM(CASE WHEN VoteTypeId = 2 THEN 1 ELSE 0 END) AS UpVotes,
            SUM(CASE WHEN VoteTypeId = 3 THEN 1 ELSE 0 END) AS DownVotes
        FROM 
            Votes
        GROUP BY 
            PostId
    ) V ON P.Id = V.PostId
    WHERE 
        P.CreationDate >= cast('2024-10-01 12:34:56' as timestamp) - INTERVAL '1 YEAR'
),
TopPosts AS (
    SELECT 
        PostId,
        Title,
        CreationDate,
        Score,
        ViewCount,
        AnswerCount,
        OwnerDisplayName,
        UpVotes,
        DownVotes
    FROM 
        RecentPosts
    WHERE 
        YearRank <= 10
),
PostDetails AS (
    SELECT 
        T.*,
        PH.PostHistoryTypeId,
        PH.CreationDate AS HistoryDate,
        PH.Comment AS CloseReason
    FROM 
        TopPosts T
    LEFT JOIN 
        PostHistory PH ON T.PostId = PH.PostId AND PH.PostHistoryTypeId IN (10, 11)
)
SELECT 
    PD.Title,
    PD.OwnerDisplayName,
    PD.CreationDate,
    PD.Score,
    PD.ViewCount,
    PD.AnswerCount,
    CASE 
        WHEN PD.CloseReason IS NOT NULL THEN 'Closed: ' || PD.CloseReason 
        ELSE 'Active' 
    END AS PostStatus,
    (SELECT COUNT(*) FROM Comments C WHERE C.PostId = PD.PostId) AS CommentCount,
    STRING_AGG(T.TagName, ', ') AS Tags
FROM 
    PostDetails PD
LEFT JOIN 
    PostLinks PL ON PD.PostId = PL.PostId
LEFT JOIN 
    Tags T ON PL.RelatedPostId = T.Id
GROUP BY 
    PD.PostId, PD.OwnerDisplayName, PD.Title, PD.CreationDate, PD.Score, PD.ViewCount, PD.AnswerCount, PD.CloseReason
ORDER BY 
    PD.Score DESC, PD.ViewCount DESC;