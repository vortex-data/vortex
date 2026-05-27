WITH RankedPosts AS (
    SELECT p.Id AS PostId, 
           p.Title, 
           p.Score, 
           p.CreationDate, 
           u.DisplayName AS Author, 
           ROW_NUMBER() OVER (PARTITION BY p.PostTypeId ORDER BY p.Score DESC) as PostRank
    FROM Posts p
    JOIN Users u ON p.OwnerUserId = u.Id
    WHERE p.CreationDate >= cast('2024-10-01 12:34:56' as timestamp) - INTERVAL '1 year'
      AND p.ViewCount > 100
),
AggregatedVotes AS (
    SELECT v.PostId, 
           SUM(CASE WHEN vt.Name = 'UpMod' THEN 1 ELSE 0 END) AS UpVotes, 
           SUM(CASE WHEN vt.Name = 'DownMod' THEN 1 ELSE 0 END) AS DownVotes
    FROM Votes v
    JOIN VoteTypes vt ON v.VoteTypeId = vt.Id
    GROUP BY v.PostId
),
TopPosts AS (
    SELECT rp.PostId, 
           rp.Title, 
           rp.Score, 
           rp.CreationDate, 
           rp.Author, 
           av.UpVotes, 
           av.DownVotes
    FROM RankedPosts rp
    LEFT JOIN AggregatedVotes av ON rp.PostId = av.PostId
    WHERE rp.PostRank <= 5 
)
SELECT tp.Title, 
       tp.Score, 
       tp.CreationDate, 
       tp.Author, 
       COALESCE(tp.UpVotes, 0) AS TotalUpVotes, 
       COALESCE(tp.DownVotes, 0) AS TotalDownVotes
FROM TopPosts tp
ORDER BY tp.Score DESC, tp.CreationDate DESC;