"""Camera mounting, markerless mat detection and which way up the mat is."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mtgtrack.engine.game import GameEngine
from mtgtrack.models.zones import Owner, Zone
from mtgtrack.vision.calibration import (
    CalibrationError,
    MatCalibration,
    calibrate_automatically,
    detect_rotation,
    find_mat_candidates,
    full_frame_calibration,
    score_quad,
)
from mtgtrack.vision.capture import FrameTransform, ListSource, TransformedSource, open_source
from mtgtrack.vision.mat import card_px
from mtgtrack.vision.orientation import read_mat_orientation, upright_score
from mtgtrack.vision.pipeline import PipelineConfig, VisionPipeline
from mtgtrack.vision.synthetic import FakeCamera, procedural_card_image

TABLES = {
    "dark wood": (38, 52, 72),
    "light wood": (120, 150, 180),
    "grey desk": (110, 110, 112),
    "black cloth": (24, 24, 24),
}


@pytest.fixture(scope="module")
def scene(demo_camera, demo_steps):
    return demo_camera.renderer.render(demo_steps[2].cards)


# ------------------------------------------------------------------ transform
def test_rotation_changes_the_frame_shape():
    frame = np.zeros((100, 200, 3), np.uint8)
    assert FrameTransform(rotate=90).apply(frame).shape[:2] == (200, 100)
    assert FrameTransform(rotate=180).apply(frame).shape[:2] == (100, 200)


def test_rotating_four_times_returns_the_original():
    frame = np.random.default_rng(0).integers(0, 255, (60, 90, 3), dtype=np.uint8)
    turned = frame
    for _ in range(4):
        turned = FrameTransform(rotate=90).apply(turned)
    assert np.array_equal(turned, frame)


def test_flip_mirrors_the_frame():
    frame = np.zeros((10, 10, 3), np.uint8)
    frame[0, 0] = 255
    assert FrameTransform(flip="h").apply(frame)[0, -1, 0] == 255
    assert FrameTransform(flip="v").apply(frame)[-1, 0, 0] == 255


def test_invalid_transforms_are_rejected():
    with pytest.raises(ValueError):
        FrameTransform(rotate=45)
    with pytest.raises(ValueError):
        FrameTransform(flip="diagonal")


def test_transform_roundtrips_and_describes_itself():
    transform = FrameTransform(rotate=270, flip="h")
    assert FrameTransform.from_dict(transform.to_dict()) == transform
    assert "270" in transform.describe()
    assert FrameTransform().describe() == "none"


def test_sources_straighten_what_they_hand_out():
    frame = np.zeros((100, 200, 3), np.uint8)
    source = TransformedSource(ListSource([frame]), FrameTransform(rotate=90))
    assert source.read().shape[:2] == (200, 100)


def test_open_source_only_wraps_when_it_has_to(tmp_path):
    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((40, 60, 3), np.uint8))
    plain = open_source(str(tmp_path))
    turned = open_source(str(tmp_path), transform=FrameTransform(rotate=90))
    assert not isinstance(plain, TransformedSource)
    assert turned.read().shape[:2] == (60, 40)


# --------------------------------------------------------------- markerless
def test_scoring_prefers_a_playmat_shaped_rectangle():
    frame_shape = (1200, 1600, 3)
    mat = np.array([[100, 100], [1500, 100], [1500, 915], [100, 915]], np.float32)
    tall = np.array([[600, 100], [900, 100], [900, 1100], [600, 1100]], np.float32)
    assert score_quad(mat, frame_shape)[0] > score_quad(tall, frame_shape)[0]


def test_a_tiny_rectangle_is_not_the_mat():
    frame_shape = (1200, 1600, 3)
    stamp = np.array([[10, 10], [60, 10], [60, 40], [10, 40]], np.float32)
    assert score_quad(stamp, frame_shape)[0] == 0.0


@pytest.mark.parametrize("table", sorted(TABLES))
def test_the_mat_is_found_on_different_tables(scene, table):
    camera = FakeCamera(seed=7, table_colour=TABLES[table])
    calibration, best = calibrate_automatically(camera.capture(scene))
    error = np.abs(np.array(calibration.src_points) - camera.corners).max()
    assert error < 25, f"{table}: off by {error:.0f}px (score {best.score:.2f})"


def test_a_mat_that_matches_the_table_is_reported_not_guessed(scene):
    """No signal is no signal -- say so rather than inventing a rectangle."""
    camera = FakeCamera(seed=7, table_colour=(56, 80, 52), table_texture=0.0)
    with pytest.raises(CalibrationError, match="rectangle|scored"):
        calibrate_automatically(camera.capture(scene))


def test_candidates_are_reported_best_first(scene):
    candidates = find_mat_candidates(FakeCamera(seed=7).capture(scene))
    assert candidates
    assert candidates == sorted(candidates, key=lambda c: -c.score)


def test_full_frame_is_always_available():
    calibration = full_frame_calibration(np.zeros((480, 640, 3), np.uint8))
    assert calibration.source == "full_frame"
    assert calibration.src_points[2] == [639, 479]


# ----------------------------------------------------------------- rotation
@pytest.mark.parametrize("mounted", [0, 90, 180, 270])
def test_a_sideways_camera_is_detected(scene, mounted):
    """Whatever rotation comes back must make the mat landscape again."""
    raw = FrameTransform(rotate=mounted).apply(FakeCamera(seed=7).capture(scene))
    correction, score = detect_rotation(raw)
    assert score > 0.5
    straightened = FrameTransform(rotate=correction).apply(raw)
    height, width = straightened.shape[:2]
    assert width > height, "the correction did not put the mat back on its side"


# -------------------------------------------------------------- orientation
def test_a_card_knows_which_way_up_it_is():
    card = procedural_card_image("Lightning Bolt")
    assert upright_score(card) > 0
    assert upright_score(cv2.rotate(card, cv2.ROTATE_180)) < 0


def test_the_cards_reveal_the_mat_orientation(scene, demo_camera):
    px = card_px(demo_camera.layout.size)
    upright = read_mat_orientation(scene, px)
    flipped = read_mat_orientation(cv2.rotate(scene, cv2.ROTATE_180), px)
    assert upright.upright and upright.certain
    assert not flipped.upright and flipped.certain


def test_an_empty_mat_admits_it_cannot_tell(demo_camera):
    blank = demo_camera.renderer.blank()
    verdict = read_mat_orientation(blank, card_px(demo_camera.layout.size))
    assert verdict.votes == 0
    assert not verdict.certain
    assert "cannot tell" in verdict.describe()


def test_turning_the_mat_around_swaps_the_ends():
    calibration = MatCalibration(
        src_points=[[0, 0], [100, 0], [100, 50], [0, 50]], mat_size=(200, 100)
    )
    turned = calibration.turned_around()
    assert turned.src_points == [[100, 50], [0, 50], [0, 0], [100, 0]]
    assert turned.turned_around().src_points == calibration.src_points


# -------------------------------------------------------------- end to end
def test_a_crooked_markerless_camera_still_tracks_the_game(
    deck, demo_camera, card_index, demo_steps
):
    """The whole point: no markers, camera mounted sideways, still correct."""
    mounted = FrameTransform(rotate=90)
    scene = demo_camera.renderer.render(demo_steps[2].cards)
    first = mounted.apply(demo_camera.camera.capture(scene))

    correction = FrameTransform(rotate=detect_rotation(first)[0])
    calibration, _ = calibrate_automatically(
        correction.apply(first), demo_camera.layout.size, transform=correction
    )
    verdict = read_mat_orientation(
        calibration.rectify(correction.apply(first)), calibration.expected_card_size()
    )
    if verdict.certain and not verdict.upright:
        calibration = calibration.turned_around()

    pipeline = VisionPipeline(
        calibration, card_index, demo_camera.layout, PipelineConfig(mask_hands=False)
    )
    engine = GameEngine(deck, card_width_px=demo_camera.card_width)
    engine.start_game()
    for step in demo_steps:
        mat = demo_camera.renderer.render(step.cards)
        for _ in range(step.frames):
            raw = mounted.apply(demo_camera.camera.capture(mat))
            engine.observe(pipeline.process(correction.apply(raw)))

    tracked = {
        (t.name, t.zone) for t in engine._previous.confirmed()
    }
    placed = {(c.name, c.zone) for c in demo_steps[-1].cards}
    assert tracked == placed


def test_zones_end_up_on_the_players_side(demo_camera, demo_steps):
    """A correctly oriented mat puts the hand at the top and lands at the bottom."""
    layout = demo_camera.layout
    hand = layout.get("player_hand").centroid(layout.size)
    lands = layout.get("player_lands").centroid(layout.size)
    assert hand[1] < lands[1]
    assert layout.zone_at(hand) == (Zone.HAND, Owner.PLAYER)
    assert layout.zone_at(lands) == (Zone.LANDS, Owner.PLAYER)
