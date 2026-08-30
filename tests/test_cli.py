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


def test_import_resolves_two_decklists(tmp_path):
    """Only the human is scanned, so only the human's deck gets an index."""
    config = Config.from_dict({"cache_dir": str(tmp_path)})
    config_path = config.save(tmp_path / "config.yaml")
    code = main([
        "-c", str(config_path), "import", str(EXAMPLES / "izzet_murktide.txt"),
        "--opponent", str(EXAMPLES / "izzet_murktide.txt"),
        "--offline", "--no-images", "--save-config",
    ])
    assert code == 0
    saved = Config.load(config_path)
    assert saved.opponent.decks == [str(EXAMPLES / "izzet_murktide.txt")]
    assert (tmp_path / "indexes" / "izzet_murktide.npz").exists()


def test_import_seats_three_ais_for_commander(tmp_path, capsys):
    config = Config.from_dict({"cache_dir": str(tmp_path)})
    config_path = config.save(tmp_path / "config.yaml")
    commander = str(EXAMPLES / "krenko_commander.txt")
    code = main([
        "-c", str(config_path), "import", commander, "--format", "commander",
        "--opponent", commander, "--opponent", commander, "--opponent", commander,
        "--offline", "--no-images", "--no-index", "--save-config",
    ])
    assert code == 0
    assert "commander: Krenko, Mob Boss" in capsys.readouterr().out
    saved = Config.load(config_path)
    assert saved.deck.format == "commander"
    assert len(saved.opponent.decks) == 3


def test_save_config_writes_back_to_the_file_it_was_given(tmp_path):
    config = Config.from_dict({"cache_dir": str(tmp_path)})
    config_path = config.save(tmp_path / "elsewhere.yaml")
    main([
        "-c", str(config_path), "import", str(EXAMPLES / "izzet_murktide.txt"),
        "--offline", "--no-images", "--no-index", "--save-config",
    ])
    assert Config.load(config_path).deck.path.endswith("izzet_murktide.txt")


def test_calibrate_finds_the_mat_without_markers(tmp_path, demo_camera, demo_steps):
    import numpy as np

    from mtgtrack.vision.calibration import MatCalibration

    config = Config.from_dict({"cache_dir": str(tmp_path)})
    config.calibration = str(tmp_path / "calib.json")
    config_path = config.save(tmp_path / "config.yaml")
    frame = demo_camera.camera.capture(demo_camera.renderer.render(demo_steps[2].cards))
    image = tmp_path / "frame.png"
    cv2.imwrite(str(image), frame)

    assert main(["-c", str(config_path), "calibrate", "--image", str(image)]) == 0
    calibration = MatCalibration.load(tmp_path / "calib.json")
    assert calibration.source == "auto"
    # The corners must be the mat's, whichever end was picked as the near one.
    found = np.array(sorted(map(tuple, calibration.src_points)))
    truth = np.array(sorted(map(tuple, demo_camera.camera.corners.tolist())))
    assert np.abs(found - truth).max() < 30

    # And the rectified mat must come out the right way up for the player.
    from mtgtrack.vision.orientation import read_mat_orientation

    verdict = read_mat_orientation(
        calibration.rectify(frame, straighten=True), calibration.expected_card_size()
    )
    assert verdict.upright


def test_calibrate_corrects_a_sideways_camera(tmp_path, demo_camera, demo_steps, capsys):
    from mtgtrack.vision.calibration import MatCalibration
    from mtgtrack.vision.capture import FrameTransform

    config = Config.from_dict({"cache_dir": str(tmp_path)})
    config.calibration = str(tmp_path / "calib.json")
    config_path = config.save(tmp_path / "config.yaml")
    upright = demo_camera.camera.capture(demo_camera.renderer.render(demo_steps[2].cards))
    crooked = FrameTransform(rotate=90).apply(upright)
    image = tmp_path / "crooked.png"
    cv2.imwrite(str(image), crooked)

    assert main(["-c", str(config_path), "calibrate", "--image", str(image)]) == 0
    out = capsys.readouterr().out
    assert "camera correction" in out
    calibration = MatCalibration.load(tmp_path / "calib.json")
    assert calibration.transform.rotate in (90, 270)
    # The stored calibration must rectify the crooked frame into a landscape mat.
    rectified = calibration.rectify(crooked, straighten=True)
    assert rectified.shape[1] > rectified.shape[0]


def test_calibrate_falls_back_to_the_whole_frame(tmp_path):
    from mtgtrack.vision.calibration import MatCalibration

    config = Config.from_dict({"cache_dir": str(tmp_path)})
    config.calibration = str(tmp_path / "calib.json")
    config_path = config.save(tmp_path / "config.yaml")
    image = tmp_path / "blank.png"
    cv2.imwrite(str(image), _blank())
    assert main([
        "-c", str(config_path), "calibrate", "--image", str(image),
        "--full-frame", "--rotate", "0",
    ]) == 0
    assert MatCalibration.load(tmp_path / "calib.json").source == "full_frame"


def test_demo_runs_the_whole_stack(tmp_path, capsys):
    config = Config.from_dict({"cache_dir": str(tmp_path)})
    config.deck.path = str(EXAMPLES / "izzet_murktide.txt")
    config_path = config.save(tmp_path / "config.yaml")
    assert main(["-c", str(config_path), "--log-level", "ERROR", "demo"]) == 0
    out = capsys.readouterr().out
    assert "scripted game" in out
    assert "final tracked state" in out
    assert "AI seats take their turns" in out


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
        engine, [BuiltinAI(deck, AIConfig(seed=1))], pipeline,
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


def test_dashboard_reports_every_seat(dashboard):
    client, _ = dashboard
    data = client.get("/api/state").json()
    assert [s["seat"] for s in data["state"]["seats"]] == [0, 1]
    assert data["opponents"][0]["seat"] == 1
    assert data["rules"]["starting_life"] == 20


def test_dashboard_runs_a_whole_ai_round(dashboard):
    client, _ = dashboard
    data = client.post("/api/turn/round").json()
    assert data["opponent_actions"]


def test_dashboard_sets_life_by_seat(dashboard):
    client, _ = dashboard
    data = client.post("/api/life/1/33").json()
    assert data["state"]["seats"][1]["life"] == 33


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
