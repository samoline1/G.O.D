## For dev without docker:

1. Run bootstrap.sh
```bash
sudo -E bash bootstrap.sh
```

2. Optional if you need a venv
```bash
python -m venv .venv || python3 -m venv .venv
```

3. Install dependencies
```bash
source .venv/bin/activate
find . -path "./venv" -prune -o -path "./.venv" -prune -o -name "requirements.txt" -exec pip install -r {} \;
./install_axolotl.sh
pip install "git+https://github.com/rayonlabs/fiber.git@1.0.0#egg=fiber[full]"
task dev_setup
task control_node_dev  # For example
```
