"""CLI, configuration and the web dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest
import yaml

from mtgtrack.cli import main
from mtgtrack.config import DEFAULT_YAML, Config

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


# ----------------------------------------------------------------- config
def test_default_yaml_parses_into_the_config():
    config = Config.from_dict(yaml.safe_load(DEFAULT_YAML))
    assert config.mat.layout == "solo"
    assert config.camera.process_fps == 8.0
    assert len(config.detector.passes) >= 3


def test_unknown_keys_are_ignored():
    config = Config.from_dict({"nonsense": 1, "camera": {"source": "5", "bogus": True}})
    assert config.camera.source == "5"


def test_robust_profile_adds_edge_passes():
    default = Config.from_dict({"detector_profile": "default"})
    robust = Config.from_dict({"detector_profile": "robust"})
    assert len(robust.detector.passes) > len(default.detector.passes)


def test_config_roundtrips_through_yaml(tmp_path):
    config = Config.from_dict({"tracker": {"min_hits": 7}, "ui": {"port": 9000}})
    again = Config.load(config.save(tmp_path / "config.yaml"))
    assert again.tracker.min_hits == 7
    assert again.ui.port == 9000


def test_index_path_follows_the_deck_name(tmp_path):
    config = Config.from_dict({"deck": {"path": "/decks/murktide.txt"}, "cache_dir": str(tmp_path)})
    assert config.index_path.name == "murktide.npz"


# -------------------------------------------------------------------- cli
def test_init_writes_a_config(tmp_path):
    target = tmp_path / "config.yaml"
    assert main(["init", "-o", str(target)]) == 0
    assert Config.load(target).mat.layout == "solo"
    # A second run must not clobber it silently.
    assert main(["init", "-o", str(target)]) == 1
    assert main(["init", "-o", str(target), "--force"]) == 0


def test_markers_produces_a_detectable_sheet(tmp_path):
    from mtgtrack.vision.calibration import detect_markers

    target = tmp_path / "markers.png"
    assert main(["markers", "-o", str(target), "--size", "200"]) == 0
    assert set(detect_markers(cv2.imread(str(target)))) == {0, 1, 2, 3}


def test_layout_dump_is_valid_json(tmp_path):
    target = tmp_path / "layout.json"
    assert main(["layout", "--dump", str(target)]) == 0
    data = json.loads(target.read_text())
    assert {r["zone"] for r in data["regions"]} >= {"hand", "lands", "battlefield"}


def test_import_resolves_a_deck_offline(tmp_path):
    config = Config.from_dict({"cache_dir": str(tmp_path)})
    config_path = config.save(tmp_path / "config.yaml")
    code = main([
        "-c", str(config_path), "import", str(EXAMPLES / "izzet_murktide.txt"),
        "--offline", "--no-images",
    ])
    assert code == 0
    assert (tmp_path / "decks" / "izzet_murktide.json").exists()
    assert (tmp_path / "indexes" / "izzet_murktide.npz").exists()


def test_calibrate_from_explicit_corners(tmp_path):
    from mtgtrack.vision.calibration import MatCalibration

    config = Config.from_dict({"cache_dir": str(tmp_path)})
    config.calibration = str(tmp_path / "calib.json")
    config_path = config.save(tmp_path / "config.yaml")
    image = tmp_path / "frame.png"
    cv2.imwrite(str(image), cv2.imread(str(image)) if image.exists() else _blank())
    code = main([
        "-c", str(config_path), "calibrate", "--image", str(image),
        "--corners", "10,10 500,12 505,300 8,298",
    ])
    assert code == 0
    assert MatCalibration.load(tmp_path / "calib.json").source == "manual"


def _blank():
    import numpy as np

    return np.full((400, 600, 3), 60, dtype="uint8")


def test_missing_decklist_is_reported_clearly(tmp_path, capsys):
    config = Config.from_dict({"cache_dir": str(tmp_path), "deck": {"path": "/nope/deck.txt"}})
    config_path = config.save(tmp_path / "config.yaml")
    assert main(["-c", str(config_path), "index", "--offline"]) == 2
    assert "decklist not found" in capsys.readouterr().out


def test_demo_runs_the_whole_stack(tmp_path, capsys):
    config = Config.from_dict({"cache_dir": str(tmp_path)})
    config.deck.path = str(EXAMPLES / "izzet_murktide.txt")
    config_path = config.save(tmp_path / "config.yaml")
    assert main(["-c", str(config_path), "--log-level", "ERROR", "demo"]) == 0
    out = capsys.readouterr().out
    assert "scripted game" in out
    assert "final tracked state" in out
    assert "opponent takes a turn" in out


# --------------------------------------------------------------- dashboard
@pytest.fixture()
def dashboard(deck, demo_camera, card_index, tmp_path):
    from fastapi.testclient import TestClient

    from mtgtrack.ai.builtin import AIConfig, BuiltinAI
    from mtgtrack.app import GameLoop, Session
    from mtgtrack.engine.game import GameEngine
    from mtgtrack.ui.web import create_app
    from mtgtrack.vision.capture import ListSource
    from mtgtrack.vision.pipeline import PipelineConfig, VisionPipeline

    config = Config.from_dict({"cache_dir": str(tmp_path)})
    pipeline = VisionPipeline(
        demo_camera.calibration, card_index, demo_camera.layout,
        PipelineConfig(mask_hands=False),
    )
    engine = GameEngine(deck, card_width_px=demo_camera.card_width)
    session = Session(
        config, deck, card_index, demo_camera.layout, demo_camera.calibration,
        engine, BuiltinAI(deck, AIConfig(seed=1)), pipeline,
    )
    loop = GameLoop(session, ListSource([]))
    loop.start_game()
    with TestClient(create_app(loop)) as client:
        yield client, loop


def test_dashboard_serves_the_page(dashboard):
    client, _ = dashboard
    response = client.get("/")
    assert response.status_code == 200
    assert "mtgtrack" in response.text


def test_dashboard_state_endpoint(dashboard):
    client, _ = dashboard
    data = client.get("/api/state").json()
    assert data["deck"]["main"] == 60
    assert data["opponent"]["engine"] == "builtin"
    assert "mana" in data


def test_dashboard_can_drive_the_opponent(dashboard):
    client, _ = dashboard
    data = client.post("/api/turn/opponent").json()
    assert data["opponent_actions"], "the opponent did nothing"


def test_dashboard_life_and_phase_controls(dashboard):
    client, _ = dashboard
    assert client.post("/api/life/player/15").json()["state"]["player"]["life"] == 15
    assert client.post("/api/phase/main1").json()["state"]["phase"] == "main1"


def test_frame_endpoint_returns_404_before_any_frame(dashboard):
    client, _ = dashboard
    assert client.get("/api/frame.jpg").status_code == 404


def test_frame_endpoint_renders_the_overlay(dashboard, demo_camera, demo_steps):
    client, loop = dashboard
    mat = demo_camera.renderer.render(demo_steps[1].cards)
    loop.step(demo_camera.camera.capture(mat))
    response = client.get("/api/frame.jpg")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 1000


def test_websocket_pushes_the_state(dashboard):
    client, _ = dashboard
    with client.websocket_connect("/ws") as socket:
        payload = json.loads(socket.receive_text())
        assert payload["state"]["player"]["library_count"] > 0
