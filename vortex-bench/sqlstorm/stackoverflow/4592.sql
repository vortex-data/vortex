
WITH UserStatistics AS (
    SELECT 
        u.Id AS UserId,
        u.DisplayName,
        u.Reputation,
        u.CreationDate,
        u.UpVotes,
        u.DownVotes,
        DENSE_RANK() OVER (ORDER BY u.Reputation DESC) AS ReputationRank,
        ROW_NUMBER() OVER (PARTITION BY u.Location ORDER BY u.Reputation DESC) AS LocationRank,
        u.Location
    FROM Users u
),
PostDetails AS (
    SELECT 
        p.Id AS PostId,
        p.OwnerUserId,
        p.Title,
        p.Score,
        p.ViewCount,
        p.CreationDate,
        COUNT(c.Id) AS CommentCount,
        SUM(CASE WHEN v.VoteTypeId = 2 THEN 1 ELSE 0 END) AS UpvoteCount,  
        SUM(CASE WHEN v.VoteTypeId = 3 THEN 1 ELSE 0 END) AS DownvoteCount  
    FROM Posts p
    LEFT JOIN Comments c ON p.Id = c.PostId
    LEFT JOIN Votes v ON p.Id = v.PostId
    GROUP BY p.Id, p.OwnerUserId, p.Title, p.Score, p.ViewCount, p.CreationDate
),
TopPosts AS (
    SELECT 
        pd.PostId,
        pd.Title,
        pd.Score,
        pd.ViewCount,
        pd.CommentCount,
        ROW_NUMBER() OVER (ORDER BY pd.Score DESC, pd.ViewCount DESC) AS ScoreRank
    FROM PostDetails pd
    WHERE pd.Score > 0
)
SELECT 
    us.DisplayName AS UserName,
    us.Reputation,
    pp.Title AS PostTitle,
    pp.Score,
    pp.ViewCount,
    pp.CommentCount,
    CASE 
        WHEN us.Location IS NULL THEN 'Unknown Location' 
        ELSE us.Location 
    END AS UserLocation,
    pht.Name AS PostHistoryType
FROM UserStatistics us
LEFT JOIN Posts p ON us.UserId = p.OwnerUserId
LEFT JOIN TopPosts pp ON p.Id = pp.PostId
LEFT JOIN PostHistory ph ON p.Id = ph.PostId
LEFT JOIN PostHistoryTypes pht ON ph.PostHistoryTypeId = pht.Id
WHERE us.ReputationRank <= 50
  AND (pp.ScoreRank <= 10 OR pp.ViewCount > 100)
  AND pp.CommentCount > COALESCE((SELECT AVG(CommentCount) FROM PostDetails), 0)
ORDER BY us.Reputation DESC, pp.Score DESC;
