"""Run the uploaded visualization functions using packaged input and output paths."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import nba_visualization as viz

BASE = Path(__file__).resolve().parent
if __name__ == '__main__':
    viz.DATA_PATH = BASE / 'data' / 'basketball_games_raw_538.csv'
    viz.BASE_DIR = BASE / 'results'
    viz.BASE_DIR.mkdir(parents=True, exist_ok=True)
    viz.main()
