import subprocess
import glob
import pathlib

# run tokenisation process
for cfg in glob.glob("configs/*.yml"):
    subprocess.run(["python", "scripts/train_w2v.py",
                    "--tokens_json", "data/tokens.json",
                    "--config", cfg])
