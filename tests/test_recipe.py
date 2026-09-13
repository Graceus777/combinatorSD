import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_accel import load_recipe, discover_characters

@pytest.mark.parametrize("value", [[], {}, {"poses": []}, {"poses": [{"lora": "pose", "weight": None}]}, {"poses": [{"lora": "pose", "weight": float("nan")}]}, {"poses": [{"lora": "pose"}], "skip_keywords": [1]}])
def test_invalid_recipes_fail_before_generation(tmp_path, value):
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        load_recipe(path)

def test_recipe_and_short_triggers(tmp_path):
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps({"poses": [{"lora": "pose", "activation": "standing"}]}), encoding="utf-8")
    assert load_recipe(recipe)["poses"][0]["lora"] == "pose"
    recipe.unlink()
    (tmp_path / "hero.json").write_text(json.dumps({"activation text": "hero", "preferred weight": 0.6}), encoding="utf-8")
    assert discover_characters(str(tmp_path))[0]["activation"] == "hero"
    assert discover_characters(str(tmp_path), ["hero"]) == []
