# NBA Basketball Prediction Phase IV User Manual

Group 3: Joshua Lutz, Latarian Lassiter and Xavier Mariscal
CST 499, North Carolina A&T State University

## What this package does

This package loads the supplied basketball_games_raw_538.csv, computes pregame
features, compares logistic regression and random forest, exports probabilities
and evaluation metrics, and generates the seven figures from the newest uploaded
visualization script. The evaluated scope is NBA seasons 2014–2023. The overall
capstone title includes WNBA, but no WNBA dataset or WNBA results are included.
The software is a command-line research prototype. It has no graphical app,
live scores, scheduled predictions or betting integration.

## Requirements and installation on Windows

1. Extract the entire ZIP to a writable folder, such as Documents\Capstone_Phase_IV.
2. Open that folder in File Explorer. Click the address bar, type `cmd`, and press Enter.
3. Check Python with `py --version`. The recorded execution used Python 3.12.14.
4. Create a virtual environment: `py -m venv .venv`.
5. Activate it in Command Prompt: `.venv\Scripts\activate`.
6. Install the tested packages: `python -m pip install -r requirements.txt`.
7. Confirm `data\basketball_games_raw_538.csv` exists. The dataset is included,
   so no data download or API key is required. Installing dependencies requires
   internet access unless those packages are already available locally.

On macOS/Linux use `python3 -m venv .venv` and `source .venv/bin/activate`.
The commands below then use `python` in that activated environment.
The Windows steps are instructions for local reproduction; the packaged run
was verified in a Linux Python environment, not on the student's Windows PC.

## Run the integrated comparison

Run `python run_phase4.py` from the extracted package folder. A successful run
prints the feature list, split counts and six model/partition metric rows.
It then writes results/model_metrics.csv, predictions.csv, baselines.csv,
run_metadata.json and phase4_confusion_matrices.png. To preserve a new console
log, run `python run_phase4.py > results\local_execution_log.txt 2>&1` on Windows.
Use an alternative CSV with `python run_phase4.py --data "C:\path\games.csv"`.
Use `--output "C:\path\outputs"` to keep a separate results directory. Repeating
the default run overwrites generated comparison outputs. The behavioral tests
read the default results folder, so run the default comparison before testing.

## Generate the original visualization results

Run `python generate_figures.py`. It uses the newest uploaded visualization
functions and redirects their input/output locations to the packaged data and
results folders. Figures are saved as PNG files instead of opening windows.
Open results in File Explorer and double-click any PNG to view it. The original
visualization model uses 2014–2021 for training and 2022–2023 for combined
validation. The Phase IV runner separates 2022 validation from 2023 testing.
Do not label the original combined validation score as the final 2023 score.

## Run tests and demonstrate the program

First run `python run_phase4.py`, then run `python test_phase4.py`.
The expected summary is `Ran 8 tests` followed by `OK`. Each test name identifies
its behavior: missing file, missing column, chronology, missing final test
season, current-game feature isolation, duplicate records, invalid probability
and metric export reconciliation. Tests create malformed CSVs in temporary
folders and remove them automatically. These examples do not alter the included
dataset. A failed test should be investigated before updating the report or
presenting a passing result to the instructor.

Start the demo by displaying this use-case list:
- Load the supplied data and confirm pregame feature generation.
- Show the chronological training, validation and test partitions.
- Compare both models using identical data and fixed settings.
- Show accuracy, log loss and confusion matrices.
- Show generated figures and exported predictions.
- Demonstrate invalid-input handling and the eight-test summary.

Then run the comparison, open model_metrics.csv and the confusion-matrix PNG,
run the figure generator, and finish with test_phase4.py. Explain that changing
a game's final score cannot change that game's pregame form, but it can update
form for later games. This is a sequential historical backtest, not a forecast
of every game made at the start of a season.

## Interpreting the recorded results

Training uses 10,193 games from 2014–2021, validation uses 1,323 games from 2022,
and testing uses 1,320 games from 2023. Logistic regression validation accuracy
is 65.61% with log loss 0.6368; its test accuracy is 63.03% with log loss 0.6481.
Random forest validation accuracy is 60.77% with log loss 0.6898; its test accuracy
is 60.30% with log loss 0.6840. Higher accuracy and lower log loss are preferable.
The fixed random forest reaches 100% training accuracy, indicating overfitting
when compared with its lower validation/test results. No test-driven tuning was
performed. Test logistic regression accuracy equals the supplied Elo baseline;
its probability log loss is slightly lower. These observed differences have
not been tested for statistical significance.

Rows of each confusion matrix represent actual outcomes; columns represent
predictions. Team 1 is class 1 and Team 2 is class 0. Team 1 is commonly the home
team in this source, but neutral-site games make team labels more precise than
calling every observation a home game. The copied original visualization uses
home/away labels. Inspect neutral-site records before interpreting those labels.

Recorded fit time was about 0.007 seconds for logistic regression and 2.880
seconds for random forest. The comparison runner took about 4.04 seconds, excluding
package installation and separate figure generation. Timings are measurements
from one environment, not hardware-independent performance guarantees.

## Troubleshooting

- File not found: extract the entire ZIP and confirm the data folder exists.
- Missing columns: supply the original raw schema, not basketball_games.csv.
- Missing partition: include seasons 2014–2021, 2022 and 2023.
- Module not found: activate the virtual environment and install requirements.
- Tests cannot find metrics: run the default comparison before the test script.
- No chart windows: expected; the program saves PNG files in results.
- Permission error: move the extracted folder into a location you can write to.
- Archived pipeline imports requests/XGBoost: those historical scripts are retained
  for reference and are not needed for the documented Phase IV run. They may
  require additional dependencies and a long walk-forward experiment.

## Package structure and maintenance

run_phase4.py coordinates comparison and exports. nba_visualization.py is a
renamed byte-for-byte copy of the newest uploaded script. generate_figures.py
runs that script with packaged paths. test_phase4.py verifies critical behavior.
originals retains all three uploaded scripts without modification. data contains
the supplied raw CSV. results contains real outputs and execution logs. The Word
report documents integration, testing and limitations. AI_PROMPTS.md records the
visible prompts and assistance in this session. DATA_NOTES.md records provenance
and scope. requirements.txt pins the versions used for this run.

## Before submitting

Capture your own program execution with Windows + Shift + S and insert the
captures where identified in the Word report. Add the actual GitHub repository
URL, a current GitHub code screenshot and a current GitHub Desktop screenshot.
Upload the current package source to that repository before capturing those
screenshots. Add prior AI coding prompts if required and record actual group/
instructor review only after it occurs. WNBA testing remains open. These items
are identified honestly in the report; this package does not establish they
were completed. Submit the Word report and the full ZIP, then prepare to run
the demonstration live.
