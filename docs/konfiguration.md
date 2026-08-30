# Konfiguration

Alles steht in einer YAML-Datei, standardmäßig
`~/.config/mtgtrack/config.yaml`. `mtgtrack init` legt sie an, `-c PFAD` wählt
eine andere. Jeder Abschnitt entspricht direkt der Datenklasse der jeweiligen
Komponente — es gibt keine zweite Wahrheit.

```yaml
camera:
  source: "0"          # Kameraindex, Videodatei, Bildordner oder RTSP-URL
  width: 1920
  height: 1080
  fps: 30
  fourcc: MJPG         # ohne MJPG liefern viele Webcams nur 5 fps bei 1080p
  autofocus: false     # aus lassen: wanderender Fokus stört die Erkennung
  process_fps: 8.0     # so oft wird ausgewertet; die Kamera darf schneller sein

mat:
  layout: solo         # solo | versus | Pfad zu einer Layout-JSON
  size: [1400, 815]    # Auflösung des entzerrten Mattenbilds
  mm: [610.0, 355.0]   # physische Mattengröße, bestimmt die erwartete Kartengröße
  recalibrate_every: 0 # >0: alle N Bilder die ArUco-Marker neu einlesen

deck:
  path: ~/decks/murktide.txt
  format: modern
  index: ""            # abweichender Pfad für den Erkennungsindex

opponent:
  engine: builtin      # builtin | forge | none
  deck: ""             # Deck der KI; leer = Spiegel deines Decks
  host: 127.0.0.1
  port: 8731
  skill: 0.85          # 0 = spielt absichtlich schlecht, 1 = so gut sie kann
  seed: null           # feste Zahl macht die KI reproduzierbar
  fallback: true

ui:
  web: true
  host: 127.0.0.1
  port: 8765
  overlay: false       # OpenCV-Debugfenster (braucht eine Desktop-Sitzung)

detector_profile: default   # default | robust
cache_dir: ""               # Standard: ~/.cache/mtgtrack
calibration: ""             # Standard: ~/.config/mtgtrack/calibration.json
log_level: INFO
```

## Feinjustage

Die folgenden Abschnitte sind optional — die Vorgaben passen für die meisten
Aufbauten. Sie sind da, wenn deine Matte oder dein Licht besonders ist.

### `detector` — Kartensuche

```yaml
detector:
  size_tolerance: 0.15      # zulässige Abweichung von der erwarteten Kartengröße
  aspect_tolerance: 0.10    # zulässige Abweichung vom Verhältnis 0,716
  min_rectangularity: 0.80  # verwirft L-förmige Verschmelzungen zweier Karten
  tap_angle_threshold: 40.0 # ab wie viel Grad eine Karte als getappt gilt
  split_merged: true        # zusammengewachsene Kartenreihen wieder auftrennen
  pad: 16                   # Rand, damit Karten am Mattenrand geschlossen wirken
  nms_iou: 0.35
```

`size_tolerance` ist bewusst eng. Die Kalibrierung kennt die Kartengröße exakt;
ein weites Fenster lässt Teile einer Karte (etwa das Artfenster) als ganze Karte
durchgehen.

### `pipeline` — Bildverarbeitung

```yaml
pipeline:
  motion_threshold: 6.0            # ab hier gilt die Matte als „in Bewegung“
  skip_detection_when_moving: true # spart viel Rechenzeit
  mask_hands: true                 # hautfarbene Flächen ausblenden
  max_distance: 0.32               # Erkennungsschwelle (kleiner = strenger)
  min_margin: 0.035                # Abstand zum Zweitplatzierten
  verify_with_orb: true            # Stichentscheid bei knappen Fällen
```

### `tracker` — zeitliche Glättung

```yaml
tracker:
  min_hits: 3              # so oft muss eine Karte gesehen werden, bevor sie zählt
  max_misses: 8            # so lange darf sie unsichtbar bleiben
  max_move_ratio: 0.75     # Suchradius in Kartenbreiten (nur für unerkannte Karten)
  min_name_agreement: 0.45 # so viel der letzten Stimmen müssen übereinstimmen
  min_confidence: 0.30
```

`min_hits` höher setzen macht die Erkennung träger, aber ruhiger. Bei
`process_fps: 5` bedeuten 3 Treffer etwa eine halbe Sekunde.

### `inference` — Spielzuglogik

```yaml
inference:
  mass_untap_threshold: 2  # so viele gleichzeitige Enttapps = neuer Zug
  infer_attacks: true      # getappte Kreaturen als Angreifer werten
  report_mana: true        # bei jeder Poolveränderung ein Ereignis senden
```

## Umgebungsvariablen

`XDG_CONFIG_HOME` und `XDG_CACHE_HOME` werden beachtet. Cache-Inhalt:

```
~/.cache/mtgtrack/
├── scryfall/cards/     aufgelöste Kartendaten (JSON)
├── scryfall/images/    heruntergeladene Kartenbilder
├── decks/              aufgelöste Decklisten
└── indexes/            Erkennungsindizes (.npz)
```

Der Cache darf jederzeit gelöscht werden; er wird beim nächsten `import` neu
aufgebaut.
