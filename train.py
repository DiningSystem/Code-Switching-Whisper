# train.py
import argparse
from runner import run_train

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    run_train(args.config)

if __name__ == "__main__":
    main()
