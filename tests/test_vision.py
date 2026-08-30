"""Calibration, detection and recognition against the synthetic mat."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mtgtrack.models.zones import Owner, Zone
from mtgtrack.vision.calibration import (
    CalibrationError,
    MatCalibration,
    calibrate_from_corners,
    calibrate_from_markers,
    detect_markers,
    generate_marker_sheet,
    order_corners,
)
from mtgtrack.vision.detect import CardDetector, warp_card
from mtgtrack.vision.mat import card_px, default_layout, solo_layout, versus_layout
from mtgtrack.vision.pipeline import PipelineConfig, VisionPipeline
from mtgtrack.vision.recognize import CardIndex, describe, hamming, phash
from mtgtrack.vision.synthetic import (
    FakeCamera,
    MatRenderer,
    PlacedCard,
    procedural_card_image,
)

SCENE = [
    ("Mountain", Zone.LANDS, True),
    ("Steam Vents", Zone.LANDS, False),
    ("Island", Zone.LANDS, False),
    ("Ragavan, Nimble Pilferer", Zone.BATTLEFIELD, False),
    ("Murktide Regent", Zone.BATTLEFIELD, True),
    ("Lightning Bolt", Zone.HAND, False),
    ("Counterspell", Zone.HAND, False),
    ("Consider", Zone.GRAVEYARD, False),
]


@pytest.fixture(scope="module")
def scene_index():
    names = [n for n, _, _ in SCENE] + ["Blood Moon", "Ponder", "Brainstorm", "Flusterstorm"]
    return CardIndex.build({n: procedural_card_image(n) for n in names})


@pytest.fixture(scope="module")
def scene():
    layout = default_layout()
    renderer = MatRenderer(layout=layout)
    cards = [
        PlacedCard(name, zone, Owner.PLAYER, tapped=tapped) for name, zone, tapped in SCENE
    ]
    return layout, renderer.render(cards)


# ------------------------------------------------------------------ hashing
def test_phash_is_stable_under_noise():
    image = procedural_card_image("Lightning Bolt")
    noisy = np.clip(
        image.astype(np.int16) + np.random.default_rng(0).normal(0, 6, image.shape), 0, 255
    ).astype(np.uint8)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey_noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
    assert hamming(phash(grey), np.array([phash(grey_noisy)]))[0] <= 6


def test_different_cards_hash_differently():
    a = describe(procedural_card_image("Lightning Bolt"))
    b = describe(procedural_card_image("Counterspell"))
    assert hamming(a.art, np.array([b.art]))[0] > 10


def test_index_recognises_its_own_reference_images(scene_index):
    for name in scene_index.names:
        result = scene_index.match(procedural_card_image(name))
        assert result.name == name, f"{name} misread as {result.name}"


def test_index_recognises_upside_down_cards(scene_index):
    flipped = cv2.rotate(procedural_card_image("Murktide Regent"), cv2.ROTATE_180)
    assert scene_index.match(flipped).name == "Murktide Regent"


def test_index_survives_a_save_load_roundtrip(scene_index, tmp_path):
    path = scene_index.save(tmp_path / "index.npz")
    again = CardIndex.load(path)
    assert set(again.names) == set(scene_index.names)
    assert again.match(procedural_card_image("Consider")).name == "Consider"


# -------------------------------------------------------------- calibration
def test_corner_ordering_is_independent_of_input_order():
    corners = [[10, 10], [110, 10], [110, 210], [10, 210]]
    for rotation in range(4):
        rotated = corners[rotation:] + corners[:rotation]
        assert order_corners(rotated) == corners


def test_calibration_maps_the_mat_onto_mat_space():
    calibration = calibrate_from_corners([[0, 0], [99, 0], [99, 199], [0, 199]], (100, 200))
    mapped = calibration.to_mat([[0, 0], [99, 199]])
    assert mapped[0] == pytest.approx([0, 0], abs=1)
    assert mapped[1] == pytest.approx([99, 199], abs=1)


def test_calibration_roundtrips_through_disk(tmp_path):
    calibration = calibrate_from_corners([[1, 2], [300, 5], [305, 210], [3, 208]])
    again = MatCalibration.load(calibration.save(tmp_path / "calib.json"))
    assert again.src_points == calibration.src_points
    assert again.mat_size == calibration.mat_size


def test_expected_card_size_follows_the_calibration():
    calibration = MatCalibration.identity((1400, 815))
    width, height = calibration.expected_card_size()
    assert width == pytest.approx(145, abs=2)
    assert height == pytest.approx(202, abs=3)


def test_marker_sheet_contains_all_four_markers(tmp_path):
    sheet = cv2.imread(str(generate_marker_sheet(tmp_path / "markers.png")))
    assert set(detect_markers(sheet)) == {0, 1, 2, 3}


def test_marker_calibration_recovers_the_mat(scene, scene_index):
    layout, mat = scene
    # A wider border so the printed markers fit beside the mat.
    camera = FakeCamera(seed=3, margin=(0.16, 0.22))
    frame = camera.capture(mat, markers=True)
    assert set(detect_markers(frame)) == {0, 1, 2, 3}

    calibration = calibrate_from_markers(frame, layout.size)
    # Markers sit outside the mat, so the recovered corners are close to but
    # not exactly the mat corners; what matters is that the result is usable.
    error = np.abs(np.array(calibration.src_points) - camera.corners).max()
    assert error < 120, f"marker calibration off by {error:.0f}px"

    pipeline = VisionPipeline(
        calibration, scene_index, layout, PipelineConfig(mask_hands=False)
    )
    names = {c.name for c in pipeline.process(frame).cards if c.name}
    assert len(names & {n for n, _, _ in SCENE}) >= 4


def test_marker_calibration_complains_when_markers_are_missing(scene):
    _, mat = scene
    with pytest.raises(CalibrationError):
        calibrate_from_markers(FakeCamera(seed=4).capture(mat), (1400, 815))


# ----------------------------------------------------------------- geometry
def test_layouts_cover_the_expected_zones():
    solo = solo_layout()
    assert {r.zone for r in solo.regions} >= {
        Zone.HAND, Zone.LANDS, Zone.BATTLEFIELD, Zone.GRAVEYARD, Zone.EXILE, Zone.LIBRARY
    }
    versus = versus_layout()
    assert {r.owner for r in versus.regions} == {Owner.PLAYER, Owner.OPPONENT, Owner.SHARED}


def test_zone_lookup_uses_the_region_polygons():
    layout = solo_layout()
    lands = layout.get("player_lands")
    assert layout.zone_at(lands.centroid(layout.size)) == (Zone.LANDS, Owner.PLAYER)
    assert layout.zone_at((-50, -50))[0] is Zone.UNKNOWN


def test_higher_priority_regions_win_overlaps():
    layout = versus_layout()
    stack = layout.get("stack")
    zone, _ = layout.zone_at(stack.centroid(layout.size))
    assert zone is Zone.STACK


# ---------------------------------------------------------------- detection
def test_detects_every_card_on_a_clean_mat(scene):
    layout, mat = scene
    detector = CardDetector(card_px(layout.size))
    assert len(detector.detect(mat)) >= len(SCENE) - 1


def test_detects_tapped_cards_as_tapped(scene):
    layout, mat = scene
    detections = CardDetector(card_px(layout.size)).detect(mat)
    tapped = [d for d in detections if d.tapped]
    assert len(tapped) >= 2


def test_warp_produces_a_portrait_card():
    image = np.zeros((400, 400, 3), np.uint8)
    quad = [[10, 10], [110, 10], [110, 150], [10, 150]]
    warped = warp_card(image, quad)
    assert warped.shape[0] > warped.shape[1]


# ------------------------------------------------------------------ pipeline
def test_pipeline_reads_the_scene_through_a_camera(scene, scene_index):
    layout, mat = scene
    camera = FakeCamera(seed=7)
    pipeline = VisionPipeline(
        camera.calibration(layout.size), scene_index, layout,
        PipelineConfig(mask_hands=False),
    )
    observation = pipeline.process(camera.capture(mat))
    seen = {(c.name, c.zone) for c in observation.cards if c.name}
    expected = {(name, zone) for name, zone, _ in SCENE}
    found = len(seen & expected)
    assert found >= len(expected) - 1, f"only found {seen}"
    assert not (seen - expected), f"hallucinated {seen - expected}"


def test_pipeline_reports_motion_and_skips_unstable_frames(scene, scene_index):
    layout, mat = scene
    camera = FakeCamera(seed=7)
    pipeline = VisionPipeline(
        camera.calibration(layout.size), scene_index, layout,
        PipelineConfig(mask_hands=False, motion_threshold=0.0),
    )
    pipeline.process(camera.capture(mat))
    second = pipeline.process(camera.capture(mat))
    assert not second.stable
    assert second.cards == []
