# Detector misreads, live session 20261127-115232 (ep0204071, detector r5)

This is a report only. No detector was trained or changed.

Sources:
- `runs/live/20261127-115232/perception.jsonl`: enemy units by class, frame and confidence. Summarised per match in `audit/enemy_classes.txt`.
- The re-read battle-log decks and plays in `reread/m<k>.json`.
- The contact sheets in `audit/sheets/`.

"True card" is the opponent's card from the battle-log deck. It is [inferred] where the deck holds several candidates. Frame numbers and counts (n = detections) are [measured].

## Misreads that reached a substantial push

| Match | Detected as | Frames (n) | True card | Effect on the policy |
|---|---|---|---|---|
| m2 | knight-hero | f1925-f3165 (n 19) | Mini P.E.K.K.A hero (played 31.7 s, 132.9 s, 155.1 s) | Seen as a weaker melee body. In E6, injecting it as a Mini PEKKA (1361 HP, dps 46) moved p(play) only from 0.109 to 0.122 over 17 decisions [measured]. So the misread was not the cause. |
| m2 | miner | f1979-f3160 (n 13); f3160 at 157.8 s on tile (9,11) | Mini P.E.K.K.A hero | Same body, split between two classes. The track was lost from f2019 to f2049. |
| m2 | hog-rider | f2014 (n 1) | Mini P.E.K.K.A hero | A one-frame flip. |
| m2 | royal-giant-ev1, cannon-ev1 | f3277-f3378 (n 15); cannon-ev1 at f2403, f3277, f3307 | Electro Giant evo | A building-targeter tank either way. The reflect stun is not in the observation under either name. |
| m3 | royal-giant-ev1 | f5042-f5218 (n 28) | Electro Giant evo | As in m2. |
| m3 | royal-hog, hog | f4078-f5167 (n 14 + 7), e.g. f4659 | Not in the deck. Probably Electro Giant evo or Bowler hero parts [inferred]. | Phantom building-targeters in the observation. |
| m8 | (lost) | f13966-f14014 | Cannon Cart (played 20.7, 81.0, 128.6, 154.4 s) | E30 is the only push whose cause is perception. Pasting the cart's own 39.5 s cell back for 40.0-44.5 s raised p(play) from 0.086 to 0.213, and P(at least one play) from 0.56 to 0.88 [measured]. |
| m8 | cannon | f13809-f14955 | Cannon Cart | Seen as a building instead of a walking troop. |
| m10 | mega-knight | f19399-f20414 (n 18), melee HP cell 3993 | Giant Skeleton (played 117.3 s) or Lumberjack evo [inferred] | The HP is about right for a Giant Skeleton. The death bomb is not represented. |
| m10 | fisherman | f19346, f20021, f20045 | Lumberjack evo | E41: the Lumberjack evo was in the observation. The policy held 6-8 elixir with answers in hand. |
| m10 | barbarian-ev1, barbarian | f20369-f20444 (n 22) | The Battle Ram evo's Barbarians | Roughly correct: the riders are Barbarians. |
| m5 | royal-recruit-ev1, royal-recruit | f9276-f9429 (n 17) | Battle Ram evo (played 180.3 s) and its riders | The ram body was read as recruits. |
| m5 | barbarian-ev1 | f9271-f9335 | The Battle Ram evo's riders | Correct class family. |
| m5 | musketeer-hero | f8059-f8725 | Magic Archer, or Electro Wizard hero [inferred] | A ranged body of about the right HP. |
| m5 | electro-wizard | f9043; also near f9061-f9079 together with ice-spirit (needs a check by eye) | Electro Wizard hero (played 79.7, 172.1 s) | The hero form was read as plain. |

## Phantom or one-off classes (not in the opponent's deck)

| Match | Class | Frames | Likely true card [inferred] |
|---|---|---|---|
| m1 | archer-queen | f748-f837 (n 11) | Wizard evo or Electro Wizard hero |
| m1 | little-prince | f1120, f1212 | Prince |
| m1 | royal-recruit | f849, f1170 | Prince |
| m1 | royal-ghost | f261 (conf 0.94) | Bandit (dash) or Skeletons evo [inferred, check by eye] |
| m1 | mega-minion, valkyrie-ev1, mother-witch, dark-prince-hero, miner | f158, f932, f889, f1236, f1265 | One-offs; see the deck in `audit/scoreboard.txt` |
| m3 | archer-queen | f4354, f4976 | Mother Witch or Bowler hero |
| m3 | fisherman | f4366, f4922 | Bowler hero |
| m3 | royal-ghost, golem, ram-rider, pekka, elite-barbarian, electro-dragon | f4813, f5340 (pekka) and single frames | One-offs |
| m4 | knight-hero | f6105-f6185 | Knight |
| m4 | royal-hog | f6210-f6240 (building-targeter cell 837 in the observation) | Not in the deck; a phantom building-targeter |
| m4 | balloon-hero | f6080-f6085 | Baby Dragon evo (air) |
| m4 | x-bow, inferno-tower, elite-barbarian | f6012, f7126, f6007, f7176 | Tombstone hero and one-offs |
| m6 | royal-ghost | f10711 (conf 0.14) | Not in the deck |
| m6 | inferno-dragon, lumberjack, lava_pup, valkyrie | f10815, f10893 | Sparky or Electro Giant evo parts |
| m6 | knight-hero | f10420 | Ice Golem hero |
| m7 | ice-golem-hero | f11843-f12994 (conf 0.18-0.60) | Goblinstein |
| m7 | giant-skeleton | f11820 | Goblinstein |
| m7 | barbarian, royal-giant, inferno-dragon, hunter, cannon | f12204-f12225, f11870, f12386, f13199 | Rascals, Cannon Cart and one-offs |
| m7 | goblin | f12625-f12805 | Possibly Goblinstein's Doctor (needs a check by eye) |
| m8 | royal-ghost | f13779-f14716 (n 7) | Not in the deck |
| m8 | golem, archer-queen | f14086, f15129, f15254 | One-offs |
| m9 | royal-ghost | f17506-f18628 (n 8) | Not in the deck |
| m9 | knight-ev1, knight-hero, knight | several | Golden Knight |
| m9 | royal-hog-ev1 | f16966 (building-targeter cell 837) | A phantom |
| m9 | royal-recruit, mega-knight, inferno-dragon | f16948, f17541, f17893-f18311 | Executioner evo, Fire Spirit and one-offs |
| m10 | inferno-dragon | f19459-f19489 (conf 0.89) | Furnace or Fire Spirit |
| m10 | royal-ghost, skeleton-dragon, mighty-miner, little-prince | f19158, f20399-f20424, f19335, f20283, f19058 | One-offs |

A note on royal-ghost: phantom royal-ghost boxes appear in m1, m3, m6, m8, m9 and m10. Only m5's deck holds the card. In m5 the real Royal Ghost evo was in the observation as a melee 1152 HP cell at conf 0.93-0.96 (64 s, 119-127 s) [measured]. A cloaked-unit row fed by phantom boxes is a risk worth checking [inferred].

## Cards the re-reader never saw (blind list)

- m2 Mini P.E.K.K.A (hero)
- m5 Battle Ram (evo)
- m6 Cannon (evo) and Ice Spirit

This is from the re-read play lists in `reread/m<k>.json`. The battle log still names these cards in the deck.

## Not detector: HUD reader faults seen in this session

- **m9: OT flag misread from t 59 to 183 s.** The live policy saw overtime during that time, including during E34. It is fixed in the audit copies only.
- **m9: elixir view read 0.0 while the HUD digit showed 2.0, at 105.9-107 s** (f16896-f16906).

## Priority (report only) [inferred]

1. Cannon Cart track loss (E30 is the only push that perception caused).
2. Hero Mini PEKKA read as knight-hero or miner. Measured as not decisive, but it is a frequent card.
3. Electro Giant evo read as royal-giant-ev1, in 3 of 10 matches.
4. Phantom royal-ghost and royal-hog boxes, which feed the cloaked and building-targeter channels.
5. Lumberjack evo read as fisherman, and Giant Skeleton as mega-knight.
