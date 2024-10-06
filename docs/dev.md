## For dev without docker:

Optional if you need a venv
```bash
python -m venv .venv || python3 -m venv .venv
```

```bash
source .venv/bin/activate
find . -path "./venv" -prune -o -path "./.venv" -prune -o -name "requirements.txt" -exec pip install -r {} \;
pip install --no-cache-dir "git+https://github.com/rayonlabs/fiber.git@1.0.0#egg=fiber[full]"
task dev_setup
task control_node_dev  # For example
```
