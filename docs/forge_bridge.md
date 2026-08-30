# Die Forge-Bridge

Die ursprüngliche Idee hinter mtgtrack war: Die Kamera erkennt, was auf dem Tisch
passiert, und reicht es an eine echte Regel-Engine weiter. Genau dafür ist die
Bridge da.

## Der ehrliche Stand

Forge hat **keine offene API und kein dokumentiertes Netzwerkprotokoll für
Fremdclients**. Es gibt keinen Weg, Forge von außen zu steuern, ohne auf der
Forge-Seite Code hinzuzufügen. Deshalb ist die Aufteilung so:

* **mtgtrack liefert** die Client-Seite, das vollständige Protokoll und einen
  lauffähigen Referenz-Server (`mtgtrack bridge-mock`), der dieselben Nachrichten
  spricht wie ein echter Adapter.
* **Noch zu bauen ist** der Forge-seitige Adapter — ein kleines Java-Programm,
  das auf dem Port lauscht, die Nachrichten in Forge-Aktionen übersetzt und
  Forges Entscheidungen zurückschickt.

Bis der Adapter existiert, ist die eingebaute KI der spielbare Gegner. Sie ist
kein Platzhalter: sie spielt ein vollständiges Deck.

## Protokoll (Version 1)

Newline-getrenntes JSON über TCP, Standardport **8731**. Eine Nachricht ist eine
Zeile UTF-8-JSON mit `v` (Protokollversion) und `type`. **Unbekannte
Nachrichtentypen müssen ignoriert werden**, nicht als Fehler behandelt — so kann
das Protokoll wachsen, ohne alte Adapter zu brechen.

### mtgtrack → Engine

```json
{"v":1,"type":"hello","deck":{...},"format":"modern"}
{"v":1,"type":"event","event":{"type":"spell_cast","card_name":"Lightning Bolt", ...}}
{"v":1,"type":"state","state":{"turn":3,"phase":"main1","player":{...},"opponent":{...}}}
{"v":1,"type":"request","what":"turn"}
{"v":1,"type":"request","what":"respond","event":{...}}
{"v":1,"type":"bye"}
```

* `hello` eröffnet die Partie und enthält die Deckliste des Gegners (die
  serialisierte Form von `Deck.to_dict()`).
* `event` ist Feuer-und-Vergessen: alles, was die Kamera erkannt hat — gezogene
  Karten, gespielte Länder, getappte Länder, Angriffe. Keine Antwort erwartet.
* `state` ist der volle Zustand zur Resynchronisation.
* `request` erwartet eine Antwort.

### Engine → mtgtrack

```json
{"v":1,"type":"ready","engine":"forge","version":"2.0"}
{"v":1,"type":"actions","actions":[
  {"kind":"play_land","card_name":"Mountain"},
  {"kind":"cast","card_name":"Lightning Bolt","targets":["Ragavan, Nimble Pilferer"]},
  {"kind":"attack","targets":["Ragavan, Nimble Pilferer"]},
  {"kind":"pass"}
]}
{"v":1,"type":"ack"}
{"v":1,"type":"error","message":"..."}
```

Gültige `kind`-Werte: `play_land`, `cast`, `attack`, `block`, `activate`, `pass`,
`mulligan`, `keep`, `concede`, `message`. Jede Aktion darf zusätzlich `text`
enthalten — eine fertige Beschreibung, die das Dashboard direkt anzeigt.

### Ablauf

```
mtgtrack                          Engine
   │  hello (Deckliste)             │
   │ ─────────────────────────────▶ │
   │             ready              │
   │ ◀───────────────────────────── │
   │  event (du ziehst Consider)    │
   │ ─────────────────────────────▶ │   (keine Antwort)
   │  event (du spielst Steam Vents)│
   │ ─────────────────────────────▶ │
   │  state + request "turn"        │
   │ ─────────────────────────────▶ │
   │            actions             │
   │ ◀───────────────────────────── │
   │  request "respond" (dein Spruch)│
   │ ─────────────────────────────▶ │
   │       actions (Counter?)       │
   │ ◀───────────────────────────── │
```

## Ausprobieren

```bash
mtgtrack bridge-mock --port 8731    # Referenz-Engine
```

In der Konfiguration:

```yaml
opponent:
  engine: forge
  host: 127.0.0.1
  port: 8731
  fallback: true      # bei Verbindungsabbruch übernimmt die eingebaute KI
```

Der Referenz-Server steckt in `src/mtgtrack/ai/forge_mock.py` und ist bewusst
kurz gehalten — er ist als ausführbare Spezifikation gedacht.

## Einen Forge-Adapter schreiben

Auf der Forge-Seite braucht es im Wesentlichen vier Dinge:

1. **Socket-Server**, der Zeilen liest und schreibt (Netty oder ein schlichter
   `ServerSocket` genügt).
2. **Deckliste übernehmen**: `hello.deck` in ein Forge-`Deck` übersetzen und eine
   Partie starten, in der Forge einen Spieler steuert.
3. **Events einspielen**: Für jedes `event` denselben Zug im Forge-Spielzustand
   nachvollziehen, damit Forges Weltbild dem Tisch entspricht. Die wichtigsten
   Typen sind `draw`, `land_played`, `spell_cast`, `permanent_entered`, `died`,
   `tapped`, `untapped` und `attack_declared`.
4. **Entscheidungen zurückgeben**: Bei `request:"turn"` Forges KI einen Zug
   spielen lassen und die ausgeführten Aktionen als `actions` serialisieren.

Praktischer Zwischenschritt, falls das zu viel ist: mtgtrack kann die Partie als
JSON-Lines-Datei mitschreiben (`export_event_log`). Damit lässt sich ein Adapter
offline entwickeln und gegen echte Mitschnitte testen, bevor er live geht.

## Eine andere Engine anbinden

Nichts am Protokoll ist forge-spezifisch. Jede Engine, die die sechs
Nachrichtentypen beantwortet, funktioniert. In Python geht das direkt über
`OpponentEngine`:

```python
from mtgtrack.ai.base import ActionKind, OpponentAction, OpponentEngine

class MeineKI(OpponentEngine):
    name = "meine-ki"

    def take_turn(self, state):
        return [OpponentAction(ActionKind.PASS)]
```
