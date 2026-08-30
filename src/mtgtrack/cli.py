"""Command line interface.

    mtgtrack init                 write a starter configuration
    mtgtrack doctor               check the camera and the installation
    mtgtrack markers -o sheet.png printable ArUco corner markers
    mtgtrack calibrate            teach it where the mat is
    mtgtrack layout --preview     check the zone layout against the mat
    mtgtrack import deck.txt      resolve a decklist and build the index
    mtgtrack run                  play
    mtgtrack demo                 play a scripted game with no camera
    mtgtrack replay frames/       re-run the pipeline over recorded frames
    mtgtrack bridge-mock          reference engine for the Forge bridge
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import __version__
from .app import (
    GameLoop,
    Session,
    SetupError,
    build_opponents,
    build_session,
    card_source,
    load_deck,
    load_layout,
    load_opponent_decks,
    open_camera,
)
from .config import DEFAULT_YAML, Config, default_config_dir
from .deck.deck import load_and_resolve
from .deck.offline import OfflineClient
from .deck.parser import parse_decklist, summarise
from .engine.game import EngineConfig, GameEngine
from .indexing import build_index
from .logging_setup import setup_logging
from .models.formats import is_known, rules_for
from .models.zones import Owner
from .vision.calibration import (
    CalibrationError,
    MatCalibration,
    calibrate_automatically,
    calibrate_from_corners,
    calibrate_from_markers,
    detect_markers,
    detect_rotation,
    find_mat_candidates,
    full_frame_calibration,
    generate_marker_sheet,
)
from .vision.capture import FrameTransform, ListSource, open_source
from .vision.mat import default_layout
from .vision.orientation import read_mat_orientation
from .vision.overlay import draw_calibration_preview, draw_layout, draw_observation
from .vision.pipeline import VisionPipeline

log = logging.getLogger(__name__)


# --------------------------------------------------------------------- utils
def _config(args: argparse.Namespace) -> Config:
    config = Config.load(getattr(args, "config", None))
    if getattr(args, "deck", None):
        config.deck.path = str(args.deck)
    if getattr(args, "source", None):
        config.camera.source = str(args.source)
    if getattr(args, "layout", None):
        config.mat.layout = str(args.layout)
    if getattr(args, "calibration", None):
        config.calibration = str(args.calibration)
    if getattr(args, "log_level", None):
        config.log_level = args.log_level
    return config


def _grab_frame(
    source_spec: str, warmup: int = 5, transform: FrameTransform | None = None
) -> np.ndarray:
    """Read one frame, discarding the first few so auto-exposure settles."""
    source = open_source(source_spec, transform=transform)
    try:
        frame = None
        for _ in range(max(1, warmup)):
            frame = source.read()
            if frame is None:
                break
        if frame is None:
            raise SetupError(f"could not read a frame from {source_spec!r}")
        return frame
    finally:
        source.release()


def _print(*parts: Any) -> None:
    print(*parts, flush=True)


# ------------------------------------------------------------------ commands
def cmd_init(args: argparse.Namespace) -> int:
    target = Path(args.output).expanduser() if args.output else default_config_dir() / "config.yaml"
    if target.exists() and not args.force:
        _print(f"{target} already exists (use --force to overwrite)")
        return 1
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(DEFAULT_YAML, encoding="utf-8")
    _print(f"wrote {target}")
    _print("next: mtgtrack markers -o markers.png && mtgtrack calibrate")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    config = _config(args)
    _print(f"mtgtrack {__version__}")
    _print(f"OpenCV     {cv2.__version__}")
    _print(f"NumPy      {np.__version__}")
    _print(f"config     {default_config_dir() / 'config.yaml'}")
    _print(f"cache      {config.cache}")

    calibration = Path(config.calibration).expanduser()
    _print(f"calibration{'':<1} {calibration} {'OK' if calibration.exists() else 'MISSING'}")
    if config.deck.path:
        deck_path = Path(config.deck.path).expanduser()
        _print(f"decklist   {deck_path} {'OK' if deck_path.exists() else 'MISSING'}")
        index = config.index_path
        _print(f"index      {index} {'OK' if index.exists() else 'not built yet'}")
    else:
        _print("decklist   (not configured)")

    _print("\ncameras:")
    found = 0
    for device in range(args.max_devices):
        capture = cv2.VideoCapture(device)
        if capture.isOpened():
            ok, frame = capture.read()
            if ok and frame is not None:
                _print(f"  [{device}] {frame.shape[1]}x{frame.shape[0]}")
                found += 1
        capture.release()
    if not found:
        _print("  none found (on Linux check that your user is in the 'video' group)")
    return 0


def cmd_markers(args: argparse.Namespace) -> int:
    path = generate_marker_sheet(args.output, marker_px=args.size)
    _print(f"wrote {path}")
    _print(
        "Print it, cut out the four markers and tape them to the mat corners:\n"
        "  id 0 top-left, id 1 top-right, id 2 bottom-right, id 3 bottom-left\n"
        "(as seen by the camera). Then run: mtgtrack calibrate"
    )
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Work out where the mat is, and which way up the camera is mounted.

    Markerless by default: the mat is the big rectangle on the table, so it can
    simply be found.  Markers, explicit corners and a whole-frame fallback are
    all still available when the table refuses to cooperate.
    """
    config = _config(args)
    size = tuple(config.mat.size)
    mm = tuple(config.mat.mm)

    raw = (
        cv2.imread(str(args.image))
        if args.image
        else _grab_frame(config.camera.source, warmup=args.warmup)
    )
    if raw is None:
        raise SetupError(f"could not read {args.image}")

    transform = _resolve_rotation(raw, args, mm)
    frame = transform.apply(raw)
    if not transform.is_identity:
        _print(f"camera correction: {transform.describe()}")

    calibration = _build_calibration(frame, args, size, mm, transform)
    _print(f"calibration source: {calibration.source}")
    calibration = _orient_mat(calibration, frame, args)

    path = calibration.save(config.calibration)
    width, height = calibration.expected_card_size()
    _print(f"written to {path}")
    _print(f"a card will measure {width:.0f}x{height:.0f} px in mat space")

    preview = args.preview or str(Path(config.calibration).with_suffix(".preview.png"))
    _write_previews(raw, frame, calibration, config, preview)
    _print(f"preview: {preview} (and *_mat.png showing the zones)")
    _print(
        "\nCheck the mat preview: your hand tray belongs at the TOP, your lands "
        "at the BOTTOM.\nIf it came out the wrong way round, re-run with "
        "--upside-down."
    )
    return 0


def _orient_mat(
    calibration: MatCalibration, frame: np.ndarray, args: argparse.Namespace
) -> MatCalibration:
    """Work out which edge of the mat the player sits at.

    A rectangle looks the same from both ends, so the cards decide it: their art
    is at the top and their text box at the bottom, which points at the player.
    Put a few cards on the mat the way you would play them and this needs no
    input at all.
    """
    if args.upside_down:
        _print("turning the mat around as requested")
        return calibration.turned_around()
    if args.no_auto_orient:
        return calibration

    verdict = read_mat_orientation(
        calibration.rectify(frame), calibration.expected_card_size()
    )
    _print(f"orientation: {verdict.describe()}")
    if verdict.votes == 0:
        _print(
            "    tip: lay a few cards face up on the mat and calibrate again, "
            "then the orientation is worked out for you"
        )
        return calibration
    if not verdict.certain:
        _print("    not certain enough to act on -- check the preview")
        return calibration
    if verdict.upright:
        return calibration
    _print("    turning the mat around so you are at the bottom")
    return calibration.turned_around()


def _resolve_rotation(
    raw: np.ndarray, args: argparse.Namespace, mm: tuple[float, float]
) -> FrameTransform:
    """Decide how the frame has to be turned before anything else happens."""
    flip = args.flip or ""
    if args.rotate == "auto":
        rotation, score = detect_rotation(raw, mm[0] / mm[1])
        if score <= 0:
            _print("could not tell which way the camera is mounted; assuming upright")
            rotation = 0
        else:
            _print(f"detected camera rotation: {rotation} deg clockwise to correct it")
        return FrameTransform(rotate=rotation, flip=flip)
    return FrameTransform(rotate=int(args.rotate), flip=flip)


def _build_calibration(
    frame: np.ndarray,
    args: argparse.Namespace,
    size: tuple[int, int],
    mm: tuple[float, float],
    transform: FrameTransform,
) -> MatCalibration:
    """Pick a calibration method: explicit corners, markers, auto, full frame."""
    if args.corners:
        return calibrate_from_corners(_parse_corners(args.corners), size, mm, transform)
    if args.full_frame:
        _print("using the whole frame as the mat")
        return full_frame_calibration(frame, size, mm, transform)

    if args.markers:
        calibration = calibrate_from_markers(frame, size, mm, transform=transform)
        _print(f"found ArUco markers {sorted(detect_markers(frame))}")
        return calibration

    # Markers are used automatically when they happen to be there, because they
    # are more precise than anything we can infer.
    try:
        calibration = calibrate_from_markers(frame, size, mm, transform=transform)
        _print(f"found ArUco markers {sorted(detect_markers(frame))}")
        return calibration
    except CalibrationError:
        pass

    candidates = find_mat_candidates(frame, mm[0] / mm[1])
    if candidates:
        _print("mat candidates (best first):")
        for candidate in candidates[:3]:
            _print(
                f"    score {candidate.score:.2f}  aspect {candidate.aspect:.2f}  "
                f"{candidate.area_fraction:.0%} of frame  via {candidate.strategy}"
                + ("  [touches the frame edge]" if candidate.touches_border else "")
            )
    calibration, best = calibrate_automatically(frame, size, mm, transform=transform)
    if best.touches_border:
        _print(
            "warning: the mat runs off the edge of the picture. Move the camera "
            "back so the whole mat is visible, or the outer zones will be cut off."
        )
    return calibration


def _write_previews(
    raw: np.ndarray,
    frame: np.ndarray,
    calibration: MatCalibration,
    config: Config,
    preview: str,
) -> None:
    """Two pictures: the detected outline, and the mat with its zones."""
    annotated = draw_calibration_preview(frame, np.array(calibration.src_points))
    Path(preview).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(preview, annotated)
    rectified = draw_layout(calibration.rectify(frame), load_layout(config))
    cv2.imwrite(str(Path(preview).with_name(Path(preview).stem + "_mat.png")), rectified)


def _parse_corners(text: str) -> list[list[float]]:
    """Four ``x,y`` pairs from the command line, in any order."""
    points = []
    for part in text.replace(";", " ").split():
        x, _, y = part.partition(",")
        points.append([float(x), float(y)])
    if len(points) != 4:
        raise SetupError("--corners needs exactly four x,y pairs")
    return points


def cmd_layout(args: argparse.Namespace) -> int:
    config = _config(args)
    layout = load_layout(config)
    if args.dump:
        Path(args.dump).write_text(json.dumps(layout.to_dict(), indent=1), encoding="utf-8")
        _print(f"layout written to {args.dump}")
    _print(f"layout '{layout.name}' at {layout.size[0]}x{layout.size[1]}")
    for region in layout.regions:
        x0, y0 = region.polygon[0]
        x1, y1 = region.polygon[2]
        _print(f"  {region.name:22} {region.zone.value:12} {region.owner.value:9} "
               f"[{x0:.2f},{y0:.2f} .. {x1:.2f},{y1:.2f}]")
    if args.preview:
        frame = (
            cv2.imread(str(args.image))
            if args.image
            else _grab_frame(config.camera.source)
        )
        calibration = MatCalibration.load(config.calibration)
        cv2.imwrite(str(args.preview), draw_layout(calibration.rectify(frame), layout))
        _print(f"preview written to {args.preview}")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    """Resolve the player's decklist, and any AI decklists, then build the index.

    Only the player is scanned, so only the player's deck needs a recognition
    index.  The AI decks are resolved for their card data alone, which is why
    importing three Commander opponents costs almost nothing.
    """
    config = _config(args)
    config.deck.format = args.format
    source = card_source(config, offline=args.offline)

    path = Path(args.decklist).expanduser()
    entries = parse_decklist(path.read_text(encoding="utf-8"))
    _print(f"parsed {len(entries)} lines: {summarise(entries)}")

    deck = load_and_resolve(path, source, name=args.name or path.stem, format=args.format)
    _print(deck.summary())
    for problem in deck.validate():
        _print(f"  ! {problem}")

    config.deck.path = str(path)
    _print(f"resolved deck cached at {deck.save(config.deck_json)}")

    opponent_paths = [str(Path(p).expanduser()) for p in (args.opponent or [])]
    for spec in opponent_paths:
        opponent_path = Path(spec)
        if not opponent_path.exists():
            raise SetupError(f"opponent decklist not found: {opponent_path}")
        opponent = load_and_resolve(
            opponent_path, source, name=opponent_path.stem, format=args.format
        )
        _print(f"opponent: {opponent.summary()}")
        for problem in opponent.validate():
            _print(f"  ! {problem}")
        opponent.save(config.deck_cache / f"{opponent_path.stem}.json")
    if opponent_paths:
        config.opponent.decks = opponent_paths
        config.opponent.deck = ""

    rules = rules_for(args.format)
    if rules.multiplayer and len(opponent_paths) < rules.default_players - 1:
        _print(
            f"note: {rules.name} seats {rules.default_players} players; "
            f"{rules.default_players - 1 - len(opponent_paths)} AI seat(s) will "
            "mirror your deck. Pass --opponent again to give them their own."
        )

    if not args.no_images and not args.offline:
        _print("downloading card art ...")
        paths = source.fetch_images(deck.unique_cards())
        _print(f"  {len(paths)}/{len(deck.unique_cards())} images available")

    if not args.no_index:
        index, report = build_index(deck, source)
        config.index_path.parent.mkdir(parents=True, exist_ok=True)
        index.save(config.index_path)
        _print(report.summary())
        _print(f"index written to {config.index_path}")

    if args.save_config:
        _print(f"configuration updated: {config.save()}")
    else:
        _print(
            "\nadd this to your config, or re-run with --save-config:"
            f"\n  deck:\n    path: {path}\n    format: {args.format}"
            + (
                "\n  opponent:\n    decks:\n"
                + "\n".join(f"      - {p}" for p in opponent_paths)
                if opponent_paths
                else ""
            )
        )
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    config = _config(args)
    deck = load_deck(config, offline=args.offline)
    index, report = build_index(deck, card_source(config, offline=args.offline))
    config.index_path.parent.mkdir(parents=True, exist_ok=True)
    index.save(config.index_path)
    _print(report.summary())
    _print(f"index written to {config.index_path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = _config(args)
    session = build_session(config, offline=args.offline, rebuild_index=args.rebuild_index)
    source = open_camera(config, session.calibration)
    loop = GameLoop(session, source)
    return _play(loop, config, overlay=args.overlay or config.ui.overlay,
                 web=not args.no_web and config.ui.web, max_frames=args.frames)


def cmd_replay(args: argparse.Namespace) -> int:
    config = _config(args)
    config.camera.source = str(args.frames_path)
    config.camera.process_fps = args.fps
    session = build_session(config, offline=args.offline)
    source = open_source(str(args.frames_path))
    loop = GameLoop(session, source)
    return _play(loop, config, overlay=args.overlay, web=args.web, max_frames=args.max_frames)


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the whole stack against a synthetic camera."""
    from .demo import DemoCamera, scripted_game

    config = _config(args)
    config.camera.process_fps = 1000.0  # the demo is not rate limited
    if getattr(args, "format", None):
        if not is_known(args.format):
            raise SetupError(f"unknown format {args.format!r}")
        config.deck.format = args.format
    if config.deck.path:
        deck = load_deck(config, offline=True)
    else:
        example = _example_decklist()
        if example is None:
            raise SetupError(
                "no deck configured and no bundled example found; "
                "point deck.path at a decklist or pass --deck"
            )
        deck = load_and_resolve(
            example, OfflineClient(), name=example.stem, format=config.deck.format
        )
    _print(deck.summary())

    camera = DemoCamera(deck, layout=default_layout(tuple(config.mat.size)), seed=args.seed)
    index, report = build_index(deck, source=None)
    _print(report.summary())

    steps = scripted_game(deck)
    frames = camera.frame_list(steps)
    labels = [label for label, _ in camera.frames(steps)]

    pipeline = VisionPipeline(
        camera.calibration, index, camera.layout, config.pipeline, config.detector
    )
    engine = GameEngine(
        deck,
        card_width_px=camera.card_width,
        config=EngineConfig(tracker=config.tracker, inference=config.inference),
        opponent_decks=load_opponent_decks(config, deck, offline=True),
    )
    opponent_decks = load_opponent_decks(config, deck, offline=True)
    opponents = build_opponents(config, opponent_decks, offline=True)
    session = Session(config, deck, index, camera.layout, camera.calibration,
                      engine, opponents, pipeline)
    loop = GameLoop(session, ListSource(frames))
    loop.start_game()
    seats = engine.state.seats
    _print(
        f"table ({deck.format}): "
        + ", ".join(f"{s.name} ({s.life})" for s in seats)
    )

    _print("\n--- scripted game ---")
    for index_, frame in enumerate(frames):
        if labels[index_]:
            _print(f"\n[{labels[index_]}]")
        for event in loop.step(frame):
            _print(f"   {event.describe()}")
    _print("\n--- the AI seats take their turns ---")
    for action in loop.opponent_round():
        _print(f"   {action.describe()}")

    state = engine.state
    _print("\n--- final tracked state ---")
    for zone in ("hand", "lands", "battlefield", "graveyard", "exile"):
        cards = [c.card.name for c in state.player.instances.values() if c.zone.value == zone]
        if cards:
            _print(f"   {zone:12} {', '.join(sorted(cards))}")
    _print(f"   library      {state.player.library_count} cards")
    _print(f"   mana         {engine.mana_pool().describe()}")
    castable = [c.card.name for c in engine.castable_from_hand()]
    _print(f"   castable     {', '.join(castable) if castable else '-'}")
    if len(state.seats) > 2:
        _print("\n--- the table ---")
        for side in state.seats:
            general = side.commander.card.name if side.commander else "-"
            _print(
                f"   seat {side.seat}  {side.name:22} life {side.life:3}  "
                f"board {len(side.battlefield()):2}  commander {general}"
            )

    if args.web:
        from .ui.web import serve

        serve(loop, config.ui.host, config.ui.port)
        _print(f"\ndashboard on http://{config.ui.host}:{config.ui.port} (ctrl-c to stop)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return 0


def _example_decklist() -> Path | None:
    """The bundled demo decklist, whether running from a checkout or installed."""
    here = Path(__file__).resolve()
    candidates = [
        parent / "examples" / "izzet_murktide.txt" for parent in here.parents[:4]
    ]
    candidates.append(Path.cwd() / "examples" / "izzet_murktide.txt")
    return next((path for path in candidates if path.exists()), None)


def cmd_bridge_mock(args: argparse.Namespace) -> int:
    from .ai.forge_mock import serve as serve_bridge

    config = _config(args)
    deck = None
    if config.deck.path:
        try:
            deck = load_deck(config, offline=args.offline)
        except SetupError as exc:
            log.warning("no default deck: %s", exc)
    _print(f"reference bridge engine on {args.host}:{args.port} (ctrl-c to stop)")
    serve_bridge(args.host, args.port, deck)
    return 0


def _play(
    loop: GameLoop, config: Config, overlay: bool, web: bool, max_frames: int | None
) -> int:
    """Shared main loop for `run` and `replay`."""
    loop.start_game()
    if web:
        from .ui.web import serve

        serve(loop, config.ui.host, config.ui.port)
        _print(f"dashboard: http://{config.ui.host}:{config.ui.port}")

    loop.on_events = lambda events: [_print(f"  {e.describe()}") for e in events]
    loop.on_opponent = lambda actions: [_print(f"  opponent {a.describe()}") for a in actions]

    if overlay:
        _print("overlay window: q quits, n ends your turn, r restarts")
    try:
        while not loop.stop_event.is_set():
            if max_frames is not None and loop.frames >= max_frames:
                break
            frame = loop.source.read()
            if frame is None:
                break
            loop.step(frame)
            if overlay and loop.last_observation is not None:
                observation = loop.last_observation
                if observation.mat_frame is not None:
                    canvas = draw_observation(
                        observation.mat_frame, observation, loop.session.layout
                    )
                    cv2.imshow("mtgtrack", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("n"):
                    event = loop.session.engine.next_turn()
                    if event.owner is Owner.OPPONENT:
                        loop.opponent_turn()
                if key == ord("r"):
                    loop.start_game()
    except KeyboardInterrupt:
        _print("\nstopping")
    finally:
        if overlay:
            cv2.destroyAllWindows()
        loop.close()
    _print(f"processed {loop.frames} frames, {len(loop.session.engine.events)} events")
    return 0


# ---------------------------------------------------------------- arg parsing
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mtgtrack",
        description="Track a physical game of Magic with an overhead camera.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"mtgtrack {__version__}")
    parser.add_argument("-c", "--config", help="path to a config file")
    parser.add_argument("--log-level", default=None,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="write a starter configuration file")
    p.add_argument("-o", "--output")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("doctor", help="check the installation and list cameras")
    p.add_argument("--max-devices", type=int, default=4)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("markers", help="generate printable ArUco corner markers")
    p.add_argument("-o", "--output", default="mtgtrack_markers.png")
    p.add_argument("--size", type=int, default=400, help="marker size in pixels")
    p.set_defaults(func=cmd_markers)

    p = sub.add_parser(
        "calibrate",
        help="work out where the mat is (no markers needed)",
    )
    p.add_argument("--source", help="camera index, video or image folder")
    p.add_argument("--image", help="calibrate from a still image instead")
    p.add_argument(
        "--rotate",
        default="auto",
        choices=["auto", "0", "90", "180", "270"],
        help="degrees clockwise to straighten a sideways camera (default: detect)",
    )
    p.add_argument("--flip", choices=["h", "v", "hv"], help="mirror the frame")
    p.add_argument("--corners", help='four "x,y" pairs, in any order')
    p.add_argument("--markers", action="store_true",
                   help="require ArUco markers instead of finding the mat")
    p.add_argument("--full-frame", action="store_true",
                   help="treat the whole picture as the mat")
    p.add_argument("--upside-down", action="store_true",
                   help="you are sitting at the other end of the detected mat")
    p.add_argument("--no-auto-orient", action="store_true",
                   help="do not work out the player's side from the cards")
    p.add_argument("--preview", help="where to write the preview images")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--calibration", help="where to write the calibration")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("layout", help="inspect or export the mat layout")
    p.add_argument("--layout", help="'solo', 'versus' or a JSON file")
    p.add_argument("--dump", help="write the layout as JSON")
    p.add_argument("--preview", help="render the zones over a camera frame")
    p.add_argument("--image", help="use this still instead of the camera")
    p.add_argument("--source")
    p.set_defaults(func=cmd_layout)

    p = sub.add_parser("import", help="resolve your decklist (and the AI's) and build the index")
    p.add_argument("decklist", help="your decklist -- the one the camera will see")
    p.add_argument(
        "--opponent",
        action="append",
        metavar="DECKLIST",
        help="an AI opponent's decklist; repeat for several (Commander)",
    )
    p.add_argument("--name")
    p.add_argument("--format", default="modern",
                   help="modern, standard, legacy, commander, ...")
    p.add_argument("--offline", action="store_true", help="use the bundled card database")
    p.add_argument("--no-images", action="store_true")
    p.add_argument("--no-index", action="store_true")
    p.add_argument("--save-config", action="store_true", help="remember this deck")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("index", help="rebuild the recognition index")
    p.add_argument("--deck")
    p.add_argument("--offline", action="store_true")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("run", help="track a live game")
    p.add_argument("--source")
    p.add_argument("--deck")
    p.add_argument("--layout")
    p.add_argument("--calibration")
    p.add_argument("--overlay", action="store_true", help="show the debug window")
    p.add_argument("--no-web", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--rebuild-index", action="store_true")
    p.add_argument("--frames", type=int, help="stop after N frames")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("replay", help="re-run the pipeline over recorded frames")
    p.add_argument("frames_path", help="video file or folder of images")
    p.add_argument("--deck")
    p.add_argument("--fps", type=float, default=1000.0)
    p.add_argument("--overlay", action="store_true")
    p.add_argument("--web", action="store_true")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--max-frames", type=int)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("demo", help="run a scripted game without a camera")
    p.add_argument("--deck")
    p.add_argument("--format", help="override the format, e.g. commander")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--web", action="store_true", help="also serve the dashboard")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("bridge-mock", help="reference engine for the Forge bridge")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8731)
    p.add_argument("--deck")
    p.add_argument("--offline", action="store_true")
    p.set_defaults(func=cmd_bridge_mock)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level or "INFO")
    try:
        return int(args.func(args) or 0)
    except SetupError as exc:
        _print(f"error: {exc}")
        return 2
    except CalibrationError as exc:
        _print(f"calibration error: {exc}")
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
