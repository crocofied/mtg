# mtgtrack

Verfolgt eine echte Magic-Partie mit einer Overhead-Kamera: die Anwendung erkennt
die Karten auf deiner Spielmatte, weiß in welcher Zone sie liegen, leitet daraus
deine Spielzüge ab — und spielt die Gegenseite, entweder mit der eingebauten KI
oder indem sie alles an eine externe Engine wie Forge weiterreicht.

**Gescannt wirst nur du.** Die Gegner sind Bots mit eigenen Decks; sie brauchen
keine physischen Karten und damit auch keine Kamera. Das ist der Grund, warum
dir die ganze Matte gehört — und warum drei zusätzliche Commander-Gegner fast
nichts kosten.

Der Kern der Idee: Sobald du deine Deckliste importierst, ist Kartenerkennung
kein offenes Problem mehr, sondern ein geschlossenes. Es müssen nicht 30.000
Karten unterschieden werden, sondern die ~22 verschiedenen Karten deiner 75.
Genau das macht die Erkennung auf einer normalen Webcam zuverlässig.

```
Kamera ─▶ Entzerrung ─▶ Kartensuche ─▶ Erkennung ─▶ Zonen ─▶ Tracking ─▶ Events ─▶ Gegner
         (Homographie)  (Kanten)      (pHash+ORB)  (Layout) (zeitlich)  (Regeln)  (KI/Forge)
```

## Was es kann

* **Ohne Kalibriermarker** — die Matte ist das große Rechteck auf dem Tisch, also
  wird sie einfach gefunden. Mehrere Segmentierungen schlagen Kandidaten vor, der
  playmat-förmigste gewinnt.
* **Schiefe Kamera** — eine seitlich montierte Kamera wird automatisch erkannt und
  geradegezogen. Welche Mattenseite deine ist, verraten die Karten selbst.
* **Zwei (oder vier) Decklisten** — deine und die der Bots. Nur dein Deck braucht
  einen Erkennungsindex.
* **Commander mit drei KIs** — 100 Karten Singleton, 40 Leben, Kommandozone,
  Commander-Steuer und Commander-Schaden.
* **Deckliste importieren** — MTGO-, Arena- und Moxfield-Format; Kartendaten und
  Bilder kommen von Scryfall und werden lokal gecacht.
* **Karten erkennen** — perceptual Hashing von Artfenster, Titelzeile und
  Gesamtkarte, ORB-Merkmalsabgleich als Stichentscheid. Getappte Karten werden
  an ihrer Drehung erkannt.
* **Zonen verfolgen** — Hand, Länder, Schlachtfeld, Stapel, Friedhof, Exil,
  Bibliothek; frei konfigurierbares Mattenlayout.
* **Spielzüge ableiten** — Ziehen, Landabwurf, Zaubersprüche, Kreaturensterben,
  Tappen, Angriffe, Zugwechsel (Massen-Enttappen = neuer Zug).
* **Mana rechnen** — welche Länder ungetappt sind, was daraus bezahlbar ist
  (inklusive Hybrid- und Phyrexianischer Kosten, per bipartitem Matching).
* **Bibliothek mitzählen** — Decklistengröße minus alles, was gesehen wurde.
* **Fehler melden** — sieht es die 5. Kopie einer 4-of, meldet es einen
  Erkennungsfehler statt still falsch weiterzurechnen.
* **Gegner spielen** — eingebaute KI (Länder, Kurve, Removal, Angriff, Block,
  Fetchländer knacken) oder Weiterleitung an Forge über eine dokumentierte
  TCP-Bridge.
* **Live-Dashboard** — Browser-Ansicht mit Zonen, Mana, Verlauf und Kamerabild.

## Was es (noch) nicht kann

Ehrlichkeitshalber vorweg:

* Es ist **keine Regel-Engine**. Der Stapel wird nicht aufgelöst, Trigger werden
  nicht verwaltet. Es beobachtet, was du tust, und protokolliert es.
* Die eingebaute KI liest Kartentext mit Mustern (Schaden, Zerstören, Ziehen,
  Countern). Was sie nicht versteht, spielt sie trotzdem — als Permanent mit
  seinen Werten. Für ernsthaftes Regelverständnis nimm die Forge-Bridge.
* **Forge hat keine offene API.** Die Bridge definiert das Protokoll und liefert
  einen lauffähigen Referenz-Server; der Forge-seitige Adapter ist noch zu
  schreiben (siehe [docs/forge_bridge.md](docs/forge_bridge.md)).
* Karten, die sich stark überlappen, werden schlechter getrennt. Lass etwa
  einen halben Zentimeter Abstand.
* Die markerlose Mattenerkennung braucht *irgendeinen* Kontrast zwischen Matte
  und Tisch. Ist beides exakt gleich, sagt sie das und du gibst die Ecken selbst
  an (`--corners`) oder nimmst `--full-frame`.
* Handkarten müssen sichtbar sein, damit sie verfolgt werden können — dafür gibt
  es die Handablage bzw. das Aufdeckfenster (siehe unten).

## Installation

```bash
git clone https://github.com/crocofied/mtg.git
cd mtg
pip install -e .
```

Für das Debug-Fenster zusätzlich `pip install -e ".[gui]"`, zum Entwickeln
`pip install -e ".[dev]"`.

## In fünf Minuten loslegen

Ohne Kamera und ohne Internet kannst du sofort sehen, was passiert:

```bash
mtgtrack demo          # spielt eine Partie auf einer synthetischen Matte durch
mtgtrack demo --web    # dasselbe, plus Dashboard auf http://127.0.0.1:8765
```

Mit echter Hardware:

```bash
mtgtrack init             # Konfigurationsdatei anlegen
mtgtrack doctor           # Kameras und Installation prüfen
mtgtrack calibrate        # Matte finden, Kameradrehung korrigieren
mtgtrack import ~/decks/meins.txt --opponent ~/decks/bot.txt --save-config
mtgtrack run              # spielen
```

Für Commander gegen drei Bots:

```bash
mtgtrack import ~/decks/meins.txt --format commander \
  --opponent ~/decks/bot1.txt \
  --opponent ~/decks/bot2.txt \
  --opponent ~/decks/bot3.txt --save-config
mtgtrack run
```

## Aufbau

Die Kamera hängt über der Matte und sieht sie vollständig. 1080p reicht; wichtiger
als Auflösung ist gleichmäßiges Licht ohne harte Schlagschatten. Autofokus
solltest du ausschalten — ein „atmender“ Fokus macht die Erkennung unruhig.

Dann einmal:

```bash
mtgtrack calibrate
```

Das war's. Kein Drucken, kein Kleben. `calibrate` erledigt drei Dinge selbst:

1. **Kameradrehung.** Hängt die Kamera um eine Vierteldrehung verkehrt, sieht die
   Matte hochkant aus. Die Drehung, die sie wieder quer macht, ist die richtige.
2. **Matte finden.** Mehrere Segmentierungen (Kanten, Helligkeit, Sättigung)
   schlagen Rechtecke vor; bewertet wird nach Größe, Konvexität, Rechtwinkligkeit
   und dem Seitenverhältnis einer Spielmatte. Der beste Kandidat gewinnt.
3. **Deine Seite.** Ein Rechteck sieht von beiden Enden gleich aus — die Karten
   nicht: Artfenster oben, Textkasten unten. Leg ein paar Karten offen auf die
   Matte und die Ausrichtung ergibt sich von selbst.

Danach schreibt es zwei Kontrollbilder (`*.preview.png` und `*_mat.png`). Im
Mattenbild gehört deine Handablage nach oben und deine Länderreihe nach unten;
stimmt das nicht, hilft `--upside-down`.

Kommt die Matte gar nicht gegen den Tisch an — gleiche Farbe, kein Rand —, sagt
das Programm das und du hast zwei Auswege: `--corners "x,y x,y x,y x,y"` oder
`--full-frame` (dann ist das ganze Bild die Matte, richte die Kamera einfach
passend aus).

ArUco-Marker sind weiter möglich und die genaueste Variante: `mtgtrack markers`
druckt sie, `calibrate` benutzt sie automatisch, sobald es sie sieht. Mit
`mat.recalibrate_every` findet sich das System nach einem Stoß gegen die Kamera
selbst wieder zurecht — mit Markern wie ohne.

Eine Standardmatte (610 × 355 mm) ist fast exakt vier Kartenhöhen tief und zehn
Kartenbreiten breit. Darauf bauen beide mitgelieferten Layouts auf:

**`solo`** — gegen die KI. Der Gegner hat keine physischen Karten, also gehört
dir die ganze Matte:

```
┌──────────────────────────────────────────────┬──────────┐
│  Handablage                                  │ Kommando │
├───────────────────────┬──────────────────────┼──────────┤
│  Stapel (Casting)     │  Aufdeckfenster      │  Exil    │
├───────────────────────┴──────────────────────┼──────────┤
│  Schlachtfeld                                │ Bibliothek│
├──────────────────────────────────────────────┼──────────┤
│  Länder                                      │ Friedhof │
└──────────────────────────────────────────────┴──────────┘
                                        ▲ du sitzt hier
```

In der rechten Spalte ganz oben sitzt die **Kommandozone** — in Modern ungenutzt,
in Commander liegt dort dein General.

**`versus`** — zwei echte Spieler: obere Hälfte Gegner, untere Hälfte du, in der
Mitte ein schmaler geteilter Streifen für Stapel und Aufdeckfenster.

Für die Hand gibt es zwei Betriebsarten:

* **Offene Hand** (empfohlen gegen die KI): Handkarten liegen offen in der
  Handablage. Beste Verfolgung, gegen einen KI-Gegner unproblematisch.
* **Aufdeckfenster**: Die Hand bleibt in der Hand. Jede gezogene Karte hältst du
  kurz über das Aufdeckfenster; die App bucht sie in die Hand und streicht sie
  wieder, sobald sie woanders auftaucht.

Eigenes Layout? `mtgtrack layout --dump layout.json`, Polygone anpassen (in
normalisierten 0–1-Koordinaten), dann `mat.layout: layout.json` setzen. Mit
`mtgtrack layout --preview zonen.png` siehst du die Zonen über dem Kamerabild.

## Befehle

| Befehl | Zweck |
| --- | --- |
| `mtgtrack init` | Startkonfiguration schreiben |
| `mtgtrack doctor` | Kameras auflisten, Installation prüfen |
| `mtgtrack calibrate` | Matte finden und Kamera geraderücken (ohne Marker) |
| `mtgtrack markers` | Optionale ArUco-Marker erzeugen |
| `mtgtrack layout` | Layout anzeigen, exportieren, über das Bild legen |
| `mtgtrack import MEINS.txt --opponent BOT.txt` | Decklisten auflösen, Bilder laden, Index bauen |
| `mtgtrack index` | Erkennungsindex neu bauen |
| `mtgtrack run` | Live-Partie verfolgen |
| `mtgtrack replay ORDNER/` | Aufgezeichnete Bilder erneut auswerten |
| `mtgtrack demo` | Vollständiger Durchlauf ohne Kamera |
| `mtgtrack bridge-mock` | Referenz-Engine für die Forge-Bridge |

## Wie es funktioniert

**Entzerrung.** Aus den vier Mattenecken wird eine Homographie berechnet, die
jedes Bild auf eine feste Draufsicht der Matte abbildet („Mat-Space“). Ab da ist
eine Karte immer gleich groß — bei 1400 px Mattenbreite exakt 145 × 202 px. Das
allein wirft fast alle Fehlkandidaten raus: Würfel, Marken, Hände und Ärmel haben
schlicht die falsche Größe.

**Kartensuche.** Kanten, Konturen, `minAreaRect`, dann Filter auf Größe,
Seitenverhältnis und Rechteckigkeit. Weil keine einzelne Kantenparametrierung
alle Karten findet — eine weiche verschmilzt Nachbarkarten, eine scharfe verliert
schwach beleuchtete — laufen mehrere Durchgänge und werden vereinigt. Zwei
Sonderfälle sind eigens behandelt: Karten am Mattenrand (das Bild wird vorher
gepolstert, sonst schließt sich ihr Umriss nicht) und zusammengewachsene
Kartenreihen (die werden an ihren Nahtlinien wieder aufgetrennt).

**Erkennung.** Jede gefundene Karte wird auf 300 × 419 px entzerrt. Verglichen
werden drei 64-Bit-DCT-Hashes (Artfenster, Titelzeile, Gesamtkarte) plus ein
Farbhistogramm — gegen den Index deines Decks, in beiden Leserichtungen, weil
der Detektor nicht weiß, wo oben ist. Liegen die zwei besten Treffer dicht
beieinander, entscheidet ORB-Merkmalsabgleich mit RANSAC.

**Tracking.** Einzelbilder lügen. Der Tracker führt pro Karte eine kurze
Historie und meldet eine Änderung erst, wenn sie mehrfach bestätigt wurde.
Zuordnung läuft **über Identität, nicht über Position**: Eine erkannte Karte darf
nur zu einem Track mit demselben Namen — egal wie weit sie gesprungen ist. Genau
das verhindert den klassischen Fehler, dass beim Nachrücken der Handkarten alle
Tracks um einen Platz verrutschen. Bewegt sich die Matte (eine Hand greift
hinein), wird der Zustand eingefroren statt aus einem verwackelten Bild
überschrieben.

**Events.** Zwei aufeinanderfolgende bestätigte Zustände werden verglichen;
jeder Zonenwechsel wird über eine Tabelle auf ein Spielereignis abgebildet
(Bibliothek→Hand = Ziehen, Hand→Länder = Landabwurf, Schlachtfeld→Friedhof =
Sterben, …). Die Zugstruktur kommt aus Heuristiken: Massen-Enttappen ist der
Enttappsegment eines neuen Zugs, getappte Kreaturen sind Angreifer.

**Verdeckte Zonen.** Was nicht sichtbar ist, wird gerechnet: Bibliothek =
Deckgröße − alles Gesehene. Und weil die Deckliste bekannt ist, kann die App
melden, wenn sie sich widerspricht.

## Gegner

**Eingebaute KI** (`opponent.engine: builtin`). Spielt ihr eigenes Deck: Mulligan
nach Landanzahl, ein Land pro Zug in der Farbe, die die Hand braucht,
Fetchländer werden geknackt, Sprüche nach Wert gecastet, Removal nur mit echtem
Ziel und passendem Permanententyp, Angriffe nach Blockermathematik, Blocks auf
Anfrage. Kreaturenwerte werden *effektiv* berechnet — ein gedrucktes 0/0, das mit
Marken ins Spiel kommt, wird nicht als harmlos eingestuft.

Am Commander-Tisch sitzt pro Gegner eine eigene Instanz mit eigener Bibliothek,
eigener Hand und eigenem General. Sie greifen sich **gegenseitig** an, nicht
reflexhaft immer dich: Ziele werden nach Bedrohung und Verteidigungslage gewählt,
mit etwas Zufall, so wie es an einem echten Tisch zugeht. Kämpfe zwischen zwei
Bots werden ausgespielt (Blocks, Trades, Lebenspunkte, Commander-Schaden) —
sonst würden sie sich ewig anstarren, ohne dass etwas passiert.

**Forge-Bridge** (`opponent.engine: forge`). Newline-getrenntes JSON über TCP.
mtgtrack schickt Deckliste, Events und Zustand, die Engine antwortet mit
Aktionen. Bricht die Verbindung ab, übernimmt die eingebaute KI, statt dass das
Spiel stehenbleibt. Zum Ausprobieren:

```bash
mtgtrack bridge-mock &            # Referenz-Engine auf Port 8731
mtgtrack run                      # mit opponent.engine: forge
```

Protokoll und Adapter-Anleitung: [docs/forge_bridge.md](docs/forge_bridge.md).

## Wenn etwas nicht erkannt wird

| Symptom | Ursache | Abhilfe |
| --- | --- | --- |
| Karten fehlen | zu wenig Kontrast | `detector_profile: robust` |
| Nachbarkarten verschmelzen | Karten liegen aneinander | ~5 mm Abstand lassen |
| Namen springen | zwei Karten sehen sich ähnlich | `mtgtrack index` meldet das kritische Paar |
| Zonen stimmen nicht | Layout passt nicht zur Matte | `mtgtrack layout --preview` |
| Alles instabil | Kamera wackelt | `mat.recalibrate_every: 30` |
| Matte nicht gefunden | Tisch wie Matte | `--corners` oder `--full-frame` |
| Zonen kopfüber | falsches Mattenende | `mtgtrack calibrate --upside-down` |
| Ruckelt | zu viele Bilder | `camera.process_fps: 5` |

Die Pipeline schafft auf einem Laptop rund 23 Bilder/s im Standardprofil und 14
im Robust-Profil — für ein Kartenspiel weit mehr als nötig.

## Entwicklung

```bash
pip install -e ".[dev]"
pytest          # 175 Tests, ~85 s, ohne Kamera und ohne Netz
ruff check src tests
```

Die Tests fahren die komplette Kette über eine synthetische Matte: gerenderte
Karten, simulierte Kameraperspektive mit Rauschen und ungleichem Licht,
Kalibrierung, Erkennung, Tracking, Events, KI und Dashboard. Deshalb lässt sich
an der Bildverarbeitung arbeiten, ohne die Kamera aufzubauen.

Weiter: [docs/architektur.md](docs/architektur.md) ·
[docs/konfiguration.md](docs/konfiguration.md) ·
[docs/forge_bridge.md](docs/forge_bridge.md)

## Lizenz

MIT. Kartendaten und Bilder stammen von [Scryfall](https://scryfall.com);
Magic: The Gathering ist eine Marke von Wizards of the Coast, die mit diesem
Projekt nichts zu tun haben.
