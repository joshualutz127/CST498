# Data notes

Input: basketball_games_raw_538.csv supplied by the user in this conversation.
The file has 73,363 rows and 14 fields, with season labels 1947–2023.
The file is identified as FiveThirtyEight NBA Elo data in the supplied code;
its remote provenance and redistribution license were not independently verified.
No remote download was performed for this package.

The selected 2014–2023 analysis contains 12,836 completed games. Earlier rows
include historical league teams beyond the modern NBA evaluation period.
The active runner selects the evaluation period before duplicate/score checks.
All raw rows are retained in the included CSV. No WNBA rows were supplied.

elo1_pre and elo2_pre are pregame ratings read directly from the supplied file;
the software does not reconstruct historical Elo ratings from an initial rating
and results. elo1_post/elo2_post and same-game final scores are excluded from
model inputs. Scores create the target and shifted past-game margin averages.
The new comparison uses elo_diff, form1 and form2, not all raw numeric fields.

The original script's non-neutral Elo formula check observed maximum absolute
error approximately 1.81e-9 with a 100-point home adjustment. That is a numerical
consistency check on this file, not independent verification of the data publisher.
The score-correlation figures are descriptive after-game analyses, not available
pregame prediction inputs. Approximate symmetry of an Elo-difference histogram
should be assessed empirically; balanced scheduling does not prove exact symmetry.

A chronological 2023 test was reserved by the added runner. Because historical
2022–2023 outputs were already examined during earlier project work, the data
is not a newly acquired blind external test set. No model tuning on 2023 was
performed in this preparation. A future-season external evaluation would provide
stronger evidence of generalization.
