
WITH UserVoteSummary AS (
    SELECT 
        UserId,
        COUNT(CASE WHEN VoteTypeId = 2 THEN 1 END) AS UpVotes,
        COUNT(CASE WHEN VoteTypeId = 3 THEN 1 END) AS DownVotes,
        COUNT(CASE WHEN VoteTypeId IN (10, 12) THEN 1 END) AS DeletedVotes
    FROM
        Votes
    GROUP BY 
        UserId
),
RecentPostActivity AS (
    SELECT 
        P.Id AS PostId,
        P.OwnerUserId,
        P.Title,
        P.CreationDate,
        P.Score,
        COALESCE(COUNT(CASE WHEN C.Id IS NOT NULL THEN 1 END), 0) AS CommentCount,
        COALESCE(COUNT(PH.Id), 0) AS EditHistoryCount
    FROM 
        Posts P
    LEFT JOIN 
        Comments C ON P.Id = C.PostId
    LEFT JOIN 
        PostHistory PH ON P.Id = PH.PostId AND PH.PostHistoryTypeId IN (4, 5, 6)
    WHERE 
        P.CreationDate > (CAST('2024-10-01 12:34:56' AS TIMESTAMP) - INTERVAL '1 year')
    GROUP BY 
        P.Id, P.OwnerUserId, P.Title, P.CreationDate, P.Score
),
RankedPosts AS (
    SELECT 
        RPA.*, 
        RANK() OVER (PARTITION BY OwnerUserId ORDER BY Score DESC) AS PostRank
    FROM 
        RecentPostActivity RPA
)
SELECT 
    UPS.UserId,
    U.DisplayName,
    U.Reputation,
    RP.PostId,
    RP.Title,
    RP.CreationDate,
    RP.Score,
    RP.CommentCount,
    RP.EditHistoryCount,
    UPS.UpVotes,
    UPS.DownVotes,
    UPS.DeletedVotes
FROM 
    UserVoteSummary UPS
JOIN 
    Users U ON UPS.UserId = U.Id
LEFT JOIN 
    RankedPosts RP ON U.Id = RP.OwnerUserId
WHERE 
    UPS.UpVotes > UPS.DownVotes
    AND RP.PostRank <= 5
    AND U.CreationDate < (CAST('2024-10-01 12:34:56' AS TIMESTAMP) - INTERVAL '6 months')
ORDER BY 
    U.Reputation DESC, 
    RP.Score DESC;
