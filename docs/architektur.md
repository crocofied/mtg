# Architektur

Fünf Schichten, jede für sich testbar. Nach unten hin gibt es keine
Rückwärtsabhängigkeiten: die Bildverarbeitung weiß nichts von Magic-Regeln, die
Regeln nichts von Kameras.

```
  models/     Karten, Zonen, Formate, Events, Zustand   (keine Abhängigkeiten)
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
* `formats.py` — eine Tabelle statt Sonderfällen: Decklegalität, Startleben,
  Kommandozone und Sitzplatzzahl pro Format.
* `gamestate.py` — `GameState` als Liste von *Sitzplätzen*. Platz 0 ist immer der
  gescannte Mensch, alle anderen sind KIs; `player` und `opponent` bleiben als
  Namen für Platz 0 und 1, was ein Zweispielerspiel nie mehr braucht.

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
* `calibration.py` — Homographie aus automatischer Mattensuche, geklickten Ecken
  oder ArUco-Markern; erkennt außerdem, wie die Kamera montiert ist. Kennt die
  physische Mattengröße und leitet daraus die erwartete Kartengröße in Pixeln ab.
* `orientation.py` — welche Mattenseite deine ist, gelesen an den Karten.
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

### Markerlose Kalibrierung in drei Schritten

Ein Rechteck zu finden ist nicht dasselbe wie eine Matte zu finden. Drei
Teilprobleme, drei Antworten:

1. **Wie ist die Kamera montiert?** Eine quer hängende Kamera zeigt die Matte
   hochkant. `detect_rotation` probiert alle vier Vierteldrehungen und nimmt die,
   die wieder eine playmat-förmige Fläche ergibt. Bei Gleichstand gewinnt die
   kleinere Drehung — ohne diese Regel würde eine gerade hängende Kamera
   willkürlich als um 180° verdreht gemeldet.
2. **Welches Rechteck ist die Matte?** `find_mat_candidates` erzeugt Kandidaten
   aus mehreren Segmentierungen (drei Canny-Schwellen, Otsu auf Grau, Sättigung
   und Helligkeit, jeweils auch invertiert) und bewertet jeden nach
   Seitenverhältnis, Rechtwinkligkeit, Parallelität und Größe. Kein einzelnes
   Verfahren funktioniert auf jedem Tisch — Kanten finden eine Matte mit klarem
   Rand, Schwellwerte eine, die mit dem Tisch verschwimmt.
3. **Welches Ende ist deins?** Geometrisch nicht entscheidbar. `orientation.py`
   liest es stattdessen an den Karten ab: der Textkasten ist heller, weniger
   gesättigt und voller waagerechter Linien, das Artfenster bunter. Über mehrere
   Karten gemittelt ist das eindeutig; sind keine Karten da, sagt es das, statt
   zu raten.

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
  Widersprüche zur Deckliste. Setzt den Tisch je nach Format: zwei Plätze für
  Modern, vier für Commander.

## ai/

`base.py` definiert `OpponentEngine`; alles andere hängt nur an dieser
Schnittstelle. `builtin.py` ist ein vollständiger Gegner — eine Instanz pro
Sitzplatz, mit eigener Bibliothek, eigener Hand und eigenem General.
`forge_bridge.py` reicht an eine externe Engine weiter und fällt bei
Verbindungsverlust auf die eingebaute KI zurück. `forge_mock.py` ist der
Referenzserver.

Zwei Entscheidungen, die den Mehrspielermodus überhaupt spielbar machen: Ziele
werden nach Bedrohung *und* Verteidigungslage gewählt statt nach „wer hat am
wenigsten Leben" (sonst wählen alle drei Bots denselben Sitz — nämlich deinen),
und Kämpfe zwischen zwei Bots werden wirklich ausgespielt (sonst bewegt sich nie
ein Lebenspunkt).

## Testansatz

Die Tests fahren die echte Kette, nicht Attrappen: gerenderte Karten werden durch
eine simulierte Kamera (Perspektive, Rauschen, ungleiches Licht) geschickt,
kalibriert, gesucht, erkannt, verfolgt und in Ereignisse übersetzt. Weil
Erkennungsindex und Renderer denselben prozeduralen Kartengenerator benutzen, ist
das ein ehrlicher Ende-zu-Ende-Test der Bildverarbeitung — nur eben ohne Kamera.

Der schärfste Test ist `test_every_scripted_board_state_is_tracked_exactly`: nach
jedem Schritt eines gescripteten Spiels muss der berichtete Zustand exakt dem
entsprechen, was auf die Matte gelegt wurde — Karte, Zone und Tapp-Status.
