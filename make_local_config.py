"""Run from the KIT directory: python make_local_config.py  -> writes config/my.local.json
Assumes: ../../my_submission, ../../apctt-env/base and ../models as in README steps 5-6."""
import json
from pathlib import Path

config = json.load(open("config/dev.json", encoding="utf-8"))
config["module"].update(
    package_id="halig-trl-assessor",
    entrypoint="participant_module.adapter:AssessmentModule",
    module_root="../../my_submission",
    package_root="../../my_submission/participant_module",
    asset_root="../../my_submission/assets",
    example_only=False,
    settings={},
)
config["checklist_path"] = "../contracts/checklist.official.json"
config["worker"]["python"] = "../../apctt-env/base/bin/python"
config["worker"]["extra_env"] = {"APCTT_MODEL_STORE": str(Path("../models").resolve())}
json.dump(config, open("config/my.local.json", "w", encoding="utf-8"), indent=2)
