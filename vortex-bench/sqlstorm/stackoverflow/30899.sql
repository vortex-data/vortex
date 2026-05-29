
WITH RecursiveTagCounts AS (
    SELECT TagName, COUNT(*) AS PostCount
    FROM Tags
    GROUP BY TagName
),
UserReputation AS (
    SELECT U.Id AS UserId, U.DisplayName, U.Reputation, 
           RANK() OVER (ORDER BY U.Reputation DESC) AS ReputationRank
    FROM Users U
),
RecentPosts AS (
    SELECT P.Id AS PostId, P.OwnerUserId, P.CreationDate, P.Title AS PostTitle, 
           PP.LastActivityDate, PT.Name AS PostTypeName, 
           ROW_NUMBER() OVER (PARTITION BY P.OwnerUserId ORDER BY P.CreationDate DESC) AS RecentPostRank
    FROM Posts P
    JOIN PostTypes PT ON P.PostTypeId = PT.Id
    LEFT JOIN Posts PP ON P.ParentId = PP.Id
    WHERE P.CreationDate > TIMESTAMP '2024-10-01 12:34:56' - INTERVAL '1 month' 
      AND P.OwnerUserId IS NOT NULL
)
SELECT 
    U.DisplayName AS AuthorDisplayName,
    U.Reputation AS AuthorReputation,
    U.ReputationRank,
    RT.TagName,
    TC.PostCount,
    RP.PostTitle,
    RP.CreationDate AS RecentPostDate,
    RP.PostTypeName,
    RP.RecentPostRank,
    COALESCE((
        SELECT COUNT(*)
        FROM Votes V
        WHERE V.PostId = RP.PostId AND V.VoteTypeId = 2
    ), 0) AS UpVotes
FROM UserReputation U
JOIN RecentPosts RP ON U.UserId = RP.OwnerUserId
JOIN PostLinks PL ON PL.PostId = RP.PostId
JOIN Tags RT ON RT.Id = PL.RelatedPostId
JOIN RecursiveTagCounts TC ON RT.TagName = TC.TagName
WHERE RP.RecentPostRank = 1
  AND RP.LastActivityDate > TIMESTAMP '2024-10-01 12:34:56' - INTERVAL '1 week'
  AND TC.PostCount > 5
ORDER BY U.Reputation DESC, TC.PostCount DESC;
