# Stage 2: MemBench-H and the difficulty check

| | |
|---|---|
| Generator | `bench/gen_membench.py` and `bench/lexicons.py` (hashes in `data/membench/manifest.json`) |
| Difficulty check | `exp/s2_difficulty.py` → `results/s2/difficulty.json`; log in `logs/s2_difficulty.log` |
| Status | Built 2026-11-04. All leakage assertions pass. Piecewise tokenisation equals natural tokenisation on all 227,507 rendered lines and questions. |

## What an episode is

An episode is a stream of user notes in prose, one line each, with no markers. The stream is followed by questions, which are answered against the final memory state.

**Entities.**
- 5–10 named persons plus the user ("my …").
- In about half the episodes, two persons share a first name, or two share a last name.

**Attributes.** 22 in all: 16 person attributes and 6 self attributes.
- **Alias families** (cell / mobile / phone number …).
- **Similar-attribute pairs:**
  - phone vs work phone;
  - email vs work email;
  - birthday vs anniversary;
  - "lives in" (city) vs "lives at" (address).

**Fact lines,** by statement family:
- explicit (alias noun);
- paraphrase or implicit ("You can reach X at …", "X moved to …");
- coreference ("I had coffee with X. She told me her email is …");
- coreference with two persons of different gender;
- multi-attribute ("X's pet is … and her door code is …");
- multi-entity ("X's cell is …, and Y's is …");
- contact cards;
- partial-name mentions ("Mr. Pham", "Arianna") once the person has been introduced.

**Changes:**
- cued updates (15% of them with a second update);
- corrections that contain both the new and the old value ("…, not OLD");
- uncued contradictions (gold = latest; conflict flag);
- deletions.

**Distractors:**
- chit-chat;
- entity mentions without facts;
- attribute mentions without values ("I should ask X for her email");
- values without entities.

**Question families:**
- canonical attribute name;
- training alias;
- paraphrase or implicit ("Where does X live?");
- partial name (first name, or "Ms. Last");
- pronoun with a preceding dialogue turn;
- pronoun with two persons of different gender in the turn;
- test-OOD only: held-out generic templates.

**Absent questions** (25% of all questions; gold = UNKNOWN):
- unknown entity;
- attribute held by no one;
- near miss (attribute held by someone else);
- similar attribute (e.g. work phone asked when only the phone is stored);
- similar name (shares a first or last name with a stored person);
- deleted fact;
- ambiguous partial name;
- mention only (attribute mentioned without a value).

**Splits.**
- Names and values are disjoint across splits (asserted: 0 overlaps in every pairing).
- **test-OOD:**
  - 60% of its fact lines use held-out statement families and held-out alias words, tagged `src_ood`;
  - all its questions use held-out question families;
  - the 75 novel words of the held-out aliases and templates occur nowhere in train or val text (asserted);
  - no 5-gram of the 69 held-out templates occurs in train or val (asserted).

| Split | Episodes | Stream tokens (mean / max) | Live + deleted records per episode | Questions | Answerable |
|---|---|---|---|---|---|
| train | 1,500 | 721 / 1,288 | 32.8 | 62,084 | 46,773 |
| val | 250 | 722 / 1,328 | 32.7 | 10,222 | 7,701 |
| test-ID | 250 | 718 / 1,177 | 32.9 | 10,395 | 7,828 |
| test-OOD | 250 | 778 / 1,304 | 32.8 | 10,404 | 7,824 |

**Template-cue check.** The answerable rate is 0.715–0.774 in every question family (test-OOD). So the phrasing does not reveal answerability.

## Difficulty check

**Setup.**
- Oracle-write mode: gold records are stored, so this tests retrieval only.
- S3 and S4 thresholds use WAEM's rule on val.

| System | test-ID exact | test-ID FAR | test-OOD exact | test-OOD FAR |
|---|---|---|---|---|
| S1: exact key string | 0.234 | 0.000 | 0.000 | 0.000 |
| S2: key-word containment | 0.361 | 0.006 | 0.207 | 0.003 |
| S2L: name + **training-alias lexicon** | 0.661 | 0.046 | 0.381 | 0.031 |
| S3: BM25 (FTS5) | 0.209 (top-1 0.834) | 0.010 | 0.114 (top-1 0.646) | 0.010 |
| S4: J-BRAIN input-embedding cosine | 0.065 | 0.014 | 0.019 | 0.009 |
| Oracle | 1.000 | 0.000 | 1.000 | 0.000 |

**Per family (S2L, test-ID).**
- canonical 1.00, alias 1.00, pronoun 1.00 (the name appears in the preceding turn);
- paraphrase 0.20; partial name 0.00.
- False acceptance on similar-attribute questions is 19.6% ("work phone" contains "phone").

**Per family (S2L, test-OOD).**
- OOD alias 0.00; paraphrase 0.05; self attributes 0.00.

**Verdict: accepted.** The best symbolic baseline is 34 points (test-ID) and 62 points (test-OOD) below the oracle, and it fails exactly on the intended phenomena. This was the acceptance condition fixed in `PLAN.md`. The generator was not revised after this check.
