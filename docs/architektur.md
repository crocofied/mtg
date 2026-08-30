# Architektur

Fünf Schichten, jede für sich testbar. Nach unten hin gibt es keine
Rückwärtsabhängigkeiten: die Bildverarbeitung weiß nichts von Magic-Regeln, die
Regeln nichts von Kameras.

```
  models/     Karten, Zonen, Events, Spielzustand      (keine Abhängigkeiten)
  deck/       Deckliste parsen, Scryfall, Offline-DB   → models
  vision/     Aufnahme, Kalibrierung, Suche, Erkennung → models
  engine/     Tracking, Mana, Inferenz, Spielzustand   → models, vision
  ai/         Gegner: eingebaut und Bridge             → models, engine
  ui/         Web-Dashboard                            → alle
  app.py      verdrahtet alles anhand der Konfiguration
```

## models/

Reine Datenklassen ohne Logik von außen.

* `card.py` — `Card` (Orakeldaten) und `CardInstance` (ein physisches Stück
  Pappe). `ManaCost` parst `{2}{U/R}{X}` inklusive Hybrid und Phyrexianisch.
* `zones.py` — `Zone` und `Owner`, plus die Aliase, die Menschen tippen.
* `events.py` — `GameEvent`, serialisierbar, mit einer lesbaren Beschreibung.
* `gamestate.py` — `GameState` mit zwei `PlayerState`.

## deck/

`parser.py` versteht MTGO, Arena, Moxfield und Abschnittsüberschriften.
`scryfall.py` löst Namen in Batches auf und cacht alles auf Platte, damit die
Kameraschleife nie auf das Netz wartet. `offline.py` liefert dieselbe
Schnittstelle aus einer mitgelieferten Kartendatenbank — deshalb laufen Demo und
Tests ohne Internet. `deck.py` prüft die Deckbaubeschränkungen (60 Karten, 15
Sideboard, maximal 4 Kopien, Basisländer ausgenommen).

## vision/

Der Kern. Die entscheidende Idee ist der **Mat-Space**: Jedes Kamerabild wird per
Homographie auf eine feste Draufsicht der Matte abgebildet. Danach ist eine Karte
immer gleich groß und immer entweder aufrecht oder um 90° gedreht — das ist der
Grund, warum die späteren Stufen so einfach bleiben dürfen.

* `capture.py` — Kamera, Videodatei, Bildordner, Speicherliste hinter einer
  Schnittstelle. Deshalb sind Tests ohne Hardware möglich.
* `calibration.py` — Homographie aus ArUco-Markern, geklickten Ecken oder
  automatischer Rechtecksuche. Kennt die physische Mattengröße und leitet daraus
  die erwartete Kartengröße in Pixeln ab.
* `mat.py` — Zonenpolygone in normalisierten Koordinaten; Punkt → Zone.
* `detect.py` — Mehrfachdurchgänge über die Kantenkarte, Konturfilter nach
  Größe/Verhältnis/Rechteckigkeit, Auftrennen verschmolzener Reihen, Entzerrung
  auf 300 × 419 px.
* `recognize.py` — drei DCT-Hashes plus Farbhistogramm gegen den Deckindex, ORB
  mit RANSAC als Stichentscheid, beide Leserichtungen.
* `pipeline.py` — Bild rein, `Observation` raus. Bewegungserkennung und
  Handmaske sitzen hier.
* `synthetic.py` — prozedurale Karten und eine simulierte Overhead-Kamera. Das
  Testfundament des Projekts.

### Warum mehrere Kantendurchgänge?

Weil keine einzelne Parametrierung alle Karten findet: Eine weiche Glättung
rettet die schlecht ausgeleuchtete Mattenecke, verschweißt aber Nachbarkarten;
eine scharfe trennt sie sauber und verliert dafür die schwachen. Gemessen fanden
verschiedene Einstellungen jeweils *andere* Karten. Also laufen mehrere
Durchgänge, jeder mit einem Rang, und ihre Ergebnisse werden vereinigt —
Treffer eines schwächeren Durchgangs überleben nur dort, wo ein besserer nichts
gefunden hat.

## engine/

* `tracker.py` — verbindet Beobachtungen über Bilder hinweg. **Identität geht vor
  Position**: Eine erkannte Karte darf nur zu einem Track mit demselben Namen,
  unabhängig von der Distanz; die Position entscheidet nur zwischen mehreren
  Kandidaten und ist allein maßgeblich, wenn eine Seite unerkannt ist.
  `TrackedState` enthält unveränderliche Momentaufnahmen — sonst verglichen zwei
  aufeinanderfolgende Zustände dieselben Objekte und der Diff wäre immer leer.
* `mana.py` — Manapool aus ungetappten Permanents; Bezahlbarkeit als bipartites
  Matching mit erweiternden Pfaden (farbige Anforderungen zuerst, generische aus
  dem Rest).
* `inference.py` — Zustandsdiff → Ereignisse, plus Zugstruktur-Heuristiken.
* `game.py` — hält den Spielzustand, rechnet verdeckte Zonen aus und meldet
  Widersprüche zur Deckliste.

## ai/

`base.py` definiert `OpponentEngine`; alles andere hängt nur an dieser
Schnittstelle. `builtin.py` ist ein vollständiger Gegner, `forge_bridge.py`
reicht an eine externe Engine weiter und fällt bei Verbindungsverlust auf die
eingebaute KI zurück. `forge_mock.py` ist der Referenzserver.

## Testansatz

Die Tests fahren die echte Kette, nicht Attrappen: gerenderte Karten werden durch
eine simulierte Kamera (Perspektive, Rauschen, ungleiches Licht) geschickt,
kalibriert, gesucht, erkannt, verfolgt und in Ereignisse übersetzt. Weil
Erkennungsindex und Renderer denselben prozeduralen Kartengenerator benutzen, ist
das ein ehrlicher Ende-zu-Ende-Test der Bildverarbeitung — nur eben ohne Kamera.

Der schärfste Test ist `test_every_scripted_board_state_is_tracked_exactly`: nach
jedem Schritt eines gescripteten Spiels muss der berichtete Zustand exakt dem
entsprechen, was auf die Matte gelegt wurde — Karte, Zone und Tapp-Status.
