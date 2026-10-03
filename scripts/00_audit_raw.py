import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from src.plts_pipeline import audit_raw, load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    audit_raw(load_config(args.config), args.run_name)
