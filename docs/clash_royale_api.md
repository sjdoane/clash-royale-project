# Clash Royale official API (developer.clashroyale.com)

Reference copied from the API's Swagger page (2026-11-18, as pasted by Sam;
the Models section was only partly expanded).

## Auth

- Tokens are JSON Web Tokens created on the developer site's My Account page.
  Each token is locked to an allow-list of public IP addresses.
- Every request sends `Authorization: Bearer API_TOKEN`.
- Base URL: `https://api.clashroyale.com/v1`.
- Tags in paths are URL-encoded, so `#ABC123` becomes `%23ABC123`.
- **Never commit a token.** Keep it in an untracked env file or the OS
  keychain, and read it from `CR_API_TOKEN`.

## Endpoints (all GET)

**clans**
- `/clans`: search clans.
- `/clans/{clanTag}`: clan information.
- `/clans/{clanTag}/members`
- `/clans/{clanTag}/warlog`
- `/clans/{clanTag}/riverracelog`
- `/clans/{clanTag}/currentwar`
- `/clans/{clanTag}/currentriverrace`

**players**
- `/players/{playerTag}`: player information.
- `/players/{playerTag}/upcomingchests`
- `/players/{playerTag}/battlelog`: list of recent battles. **This is the one
  the project uses:** decks, levels, crowns and the result for each battle,
  including top players'.

**cards**
- `/cards`: list of available cards.

**tournaments**
- `/tournaments`: search.
- `/tournaments/{tournamentTag}`

**locations and rankings**
- `/locations`
- `/locations/{locationId}`
- `/locations/{locationId}/rankings/clans`
- `/locations/{locationId}/rankings/players`
- `/locations/{locationId}/rankings/clanwars`
- `/locations/{locationId}/pathoflegend/players`: Path of Legend rankings for a location.
- `/locations/global/pathoflegend/{seasonId}/rankings/players`: top Path of
  Legend players for a season. The model is PlayerPathOfLegendRanking:
  tag, name, expLevel, rank, eloRating, clan.
- `/locations/global/seasons`
- `/locations/global/seasonsV2`: seasons with more detail.
- `/locations/global/seasons/{seasonId}`
- `/locations/global/seasons/{seasonId}/rankings/players`
- `/locations/global/rankings/tournaments/{tournamentTag}`

**events and leaderboards**
- `/events`: current events.
- `/leaderboards`
- `/leaderboard/{leaderboardId}`: entries have name, tag, score, rank,
  updated and clan.
- `/leaderboard/{leaderboardId}/leagues/{leagueId}`
- `/globaltournaments`

## Models (as far as expanded)

- **LeaderboardEntry**: clan (PlayerClan), name, tag, score, rank, updated.
- **PlayerClan**: badgeId, tag, name, badgeUrls.
- **PlayerPathOfLegendRanking**: clan (PlayerRankingClan), tag, name,
  expLevel, rank, eloRating.
- **PlayerRankingClan**: badgeId, tag, name, badgeUrls.
- **LadderTournamentRanking**: clan, wins, losses, tag, name, rank, previousRank.
- **CurrentRiverRace**: state (enum, 6 values), clan (RiverRaceClan), clans,
  collectionEndTime, warEndTime, sectionIndex, periodIndex, periodType
  (enum, 3 values), periodLogs.
- **PeriodLog**: items (PeriodLogEntry list), periodIndex.
- **PeriodLogEntry**: clan {tag}, pointsEarned, progressStartOfDay,
  progressEndOfDay, endOfDayRank, progressEarned, numOfDefensesRemaining,
  progressEarnedFromDefenses.
- **RiverRaceClan**: badgeId, clanScore, name, tag, fame, repairPoints,
  finishTime, participants, periodPoints.
- **RiverRaceParticipant**: tag, name, fame, repairPoints, boatAttacks,
  decksUsed, decksUsedToday.

Not expanded in the paste. Check the live Swagger before relying on these:

Location, Replay, Version, JsonNode, LeagueSeason(List), LocationList,
Fingerprint, Leaderboard(List), JsonLocalizedName, ClanMember(List), Arena,
LadderTournament(List), SurvivalMilestoneReward(List), Item, GameMode,
PlayerRanking(List), VerifyTokenRequest/Response, Clan, Match, **Battle**,
**PlayerBattleData(List)**, PlayerBattleAugment(List), PlayerBattleRound(List),
PlayerItemLevel(List), CancelMatchResponse, **BattleList**, CurrentClanWar,
ClanWarClan(List), ClanWarParticipant(List), RiverRaceLog(Entry),
RiverRaceStanding(List), Tournament, TournamentMember(List),
TournamentHeader(List), Emote(List), ClanList, ClanRanking(List),
RegisterMatchRequest/Response, UpcomingChests, Chest(List), ClanWarLog(Entry),
ClanWarStanding(List), GameModeDetails(List), ChaosGameModeModifier(List),
**Player**, PathOfLegendSeasonResult, PlayerAchievementProgress(List),
PlayerAchievementBadge(List), PlayerLeagueStatistics, LeagueSeasonResult,
TrailEvent(List), Items, ItemList, ClientError.

There is a `Replay` model, but no endpoint above returns replays.

## Planned uses

- **Opponent deck pool:** pull the current Path of Legend top players, then
  their battle logs, and count the decks they face. This replaces the stale
  `python_ai/opponents/decks/meta_decks.json`.
- **Video labels:** match each video's battles to the uploader's battle log by
  time and opponent name, to get the opponent's full deck, levels and the
  result. The battle log only keeps recent battles (about 25), so pull it
  regularly. The video's channel name is not the player tag, so look up the
  tag once per player.
