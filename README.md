# Funkmesh Repeater-Simulator

Open-Source-Werkzeug von **Funkmesh Deutschland e.V.** zur Planung und Prüfung der Funkabdeckung von
**MeshCore-Repeatern** (LoRa, 868 MHz) – für **ganz Deutschland**. Man wählt ein Gebiet (Kreis, Stadt,
Bundesland, eigenes Rechteck oder ganz Deutschland), der Simulator lädt die passenden Geodaten und
berechnet die Abdeckung:

* **in NRW** bis auf einzelne Straßenzüge aus amtlichen 1-m-Laserdaten: Gelände (DGM1), 3D-Gebäude mit
  Dachformen (LoD2), Bewuchs (DOM1 − DGM1);
* **im übrigen Deutschland** aus Copernicus-Höhenmodell (30 m), ESA-Landbedeckung (10 m) und
  OpenStreetMap-Gebäuden.

Läuft lokal unter **Linux und Windows**, nutzt eine **NVIDIA-GPU** (CUDA) und rechnet ohne GPU auf allen
CPU-Kernen.

## Funktionen

* **Gebiete** (Reiter „Gebiet“): Kreis/Stadt/Bundesland suchen (Grenze von OpenStreetMap), Rechteck
  in die Karte ziehen, aktuellen Kartenausschnitt oder **ganz Deutschland** wählen. Vorab gibt es eine
  Abschätzung von Fläche, Datenquellen, Download, Plattenplatz, Vorbereitungs- und Rechenzeit; bei
  großen Gebieten zusätzlich eine deutliche Warnung und Rückfrage. Die Vorbereitung läuft im
  Hintergrund, lässt sich anhalten und fortsetzen. Jedes Gebiet hat eigene Repeater, Parameter,
  Ergebnisse und Kalibrierung; ein neues Gebiet übernimmt Parameter und die darin liegenden Repeater
  des aktiven Gebiets.
* Repeater auf der Karte setzen und verschieben. Einstellbar sind Antennenhöhe über Grund,
  Sendeleistung, Antennengewinn, Kabelverlust und Downtilt. Beim Setzen werden Gelände-, Dach- und
  Bewuchshöhe am Standort angezeigt.
* **Import der Repeater von [map.meshcore.io](https://map.meshcore.io) und [mapme.sh](https://mapme.sh)**
  (einzeln oder beide zusammen, nach Public Key zusammengeführt): Filter nach Alter, bereits
  vorhandene Repeater werden erkannt.
* **Mehrfachauswahl**: Repeater markieren (alle, gefiltert, im Kartenausschnitt, inaktive) und
  gemeinsam löschen, (de)aktivieren oder ihre Antennenhöhe setzen.
* **Zurücksetzen** (Reiter „Parameter“): setzt das aktive Gebiet komplett auf Anfang (Repeater,
  Parameter, Ergebnisse, Auslastung, Kalibrierung, Messdaten); die Geodaten bleiben.
* Die **Ausleuchtung** wird erst auf Knopfdruck berechnet. Jeder Repeater wird gecacht, nach dem Setzen
  eines neuen Repeaters wird also nur dieser gerechnet.
* Kartenebenen:
  * Linkreserve (min. aus Down- und Uplink)
  * Versorgungswahrscheinlichkeit
  * Empfangspegel
  * Redundanz (Anzahl der Repeater)
  * bester Repeater
  * Kanalbelegung
  * Gelände, Gebäude, Bewuchs
* **Punktabfrage**: Ein Klick in die Karte zeigt Pegel, Reserve und alle hörbaren Repeater.
* **Höhenprofil** Repeater → Punkt mit Gebäuden, Bewuchs, Sichtlinie und Fresnelzone.
* **Repeater-zu-Repeater-Verbindungen** (Mesh-Topologie) mit Linkreserve.
* **Auslastungssimulation**: Duty Cycle, Kanalbelegung, Kollisionen und Flood-Reichweite.
* **Kalibrierung mit eigenen Messungen**: Messdateien (CSV/JSON der Mapper-App oder eigene CSV)
  einfach in den Reiter ziehen – mehrere Dateien werden gesammelt, gemeinsam ausgewertet und das
  Modell wird automatisch angepasst, wenn das Ergebnis plausibel ist.

## Installation

Voraussetzung: Python 3.10 bis 3.13. Für die GPU-Beschleunigung braucht es zusätzlich einen
NVIDIA-Treiber; das CUDA-Toolkit wird per pip mitinstalliert.

Die Mess-App liegt als Git-Submodul in `funkmesh_sim/mapper` (eigenes Repo
[meshcore-mapper](https://github.com/funkmesh-deutschland/meshcore-mapper)), deshalb mit
`--recurse-submodules` klonen:

```bash
git clone --recurse-submodules https://github.com/funkmesh-deutschland/meshcore-repeater-simulator.git
cd meshcore-repeater-simulator
```

Bei einem schon vorhandenen Klon ohne Submodul: `git submodule update --init`.

**Linux**

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[gpu]"
```

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\pip install -e ".[gpu]"
```

Ohne NVIDIA-GPU `.[gpu]` weglassen; dann wird auf der CPU gerechnet. Mit `FUNKMESH_NO_GPU=1` lässt
sich die CPU erzwingen.

## Nutzung

**Oberfläche starten** und im Browser <http://127.0.0.1:8765> öffnen:

```bash
.venv/bin/python -m funkmesh_sim serve
```

Beim ersten Start gibt es noch kein Gebiet: Im Reiter **Gebiet** einen Kreis suchen (oder Rechteck
ziehen / ganz Deutschland), Abschätzung prüfen, **Gebiet vorbereiten**. Danach Repeater importieren
und **Ausleuchtung berechnen**.

Richtwerte (gemessen, RTX 4070, 100-Mbit-Leitung):

| Gebiet | Raster | Daten | Vorbereitung | Platte | Ausleuchtung |
|---|---|---|---|---|---|
| Kreis Coesfeld (NRW) | 5 m | NRW 1 m, ~25 GB Download | ~1 h (Schätzung) | 1 GB | 37 Repeater ~3,5 min |
| Stadt Gifhorn (NI) | 10 m | Copernicus + OSM | 3 min | 30 MB | 16 Repeater 22 s |
| Saarland | 50 m | Copernicus, Clutter | 7 s | 20 MB | 204 Repeater + 15 500 Verbindungen 2 min |
| ganz Deutschland | 50 m | Copernicus, Clutter | ~25 min (Schätzung) | 2 GB | ~6 500 Repeater: einige Stunden (Schätzung) |

Gebiete lassen sich auch per Kommandozeile vorbereiten (fortsetzbar):

```bash
.venv/bin/python -m funkmesh_sim prepare kreis-borken --bbox 6.4,51.7,7.2,52.2 --res 10
.venv/bin/python -m funkmesh_sim prepare deutschland --bbox de --res 50
.venv/bin/python -m funkmesh_sim areas
```

Unter Windows jeweils `.venv\Scripts\python` statt `.venv/bin/python` verwenden.

Arbeitsverzeichnis für Daten, Caches und Projekte ist `./data` (über `FUNKMESH_DATA` änderbar).

### Speicher (kein OOM)

* Alle großen Raster liegen als Memory-Map auf der Platte.
* Vor jeder großen Allokation werden RAM und VRAM geprüft. Standardgrenzen:
  * RAM: höchstens 35 % des freien RAM und höchstens 6 GiB
  * VRAM: höchstens 60 % des freien VRAM
* Anpassbar über `FUNKMESH_RAM_FRACTION`, `FUNKMESH_RAM_MAX_GB` und `FUNKMESH_VRAM_FRACTION`.
* Reicht das Budget nicht, bricht die Berechnung mit einer Meldung ab, statt den Rechner einzufrieren.
* Vor dem Anlegen eines Gebiets wird der freie Plattenplatz geprüft (Raster × 1,6 + 5 GiB Reserve).
* Große Gebiete werden in Blöcken zu 1024 × 1024 Zellen aufbereitet; Repeater-Verbindungen werden in
  Paketen zu 1000 Strecken gerechnet.

## Mess-App „Funkmesh Mapper“ (PWA fürs Tablet)

Ordner `funkmesh_sim/mapper/` (Submodul, eigenes Repo, veröffentlicht unter
<https://funkmesh-deutschland.github.io/meshcore-mapper/>); der Simulator liefert sie zusätzlich unter `/mapper/` aus. Die App ist rein
statisch: kein Server-Code, alle Daten bleiben auf dem Gerät (IndexedDB), bis man sie exportiert.

* **Verbindung** zum MeshCore-Companion per Web Bluetooth oder USB (Web Serial). Grundlage ist
  [meshcore.js](https://github.com/meshcore-dev/meshcore.js) (MIT, unter `vendor/` eingebunden).
* **Passive Erfassung**: Der Companion meldet jedes empfangene Paket mit RSSI und SNR (`LogRxData`).
  Zugeordnet wird so:
  * Flood-Paket: der letzte Pfad-Hash (1–3 Byte) ist der direkt gehörte Repeater.
  * Repeater-Advert ohne Hop: voller Public Key.
  * Direkt geroutete Pakete sind nicht eindeutig und werden nicht gespeichert.
* **Aktiv-Test (optional)**: alle N Sekunden ein Trace über 1 Hop an gewählte Repeater, ohne Flooding.
  * Antwort: Downlink-RSSI und Uplink-SNR am Repeater.
  * Keine Antwort: Zeile „nicht empfangen“.
* **Manuell**: Pegel oder „kein Empfang“ von Hand, nur mit aktuellem GPS-Fix.
* Jede Zeile enthält Zeit, GPS (mit Genauigkeit und Alter), Höhe und Messaufbau (zu Fuß / Fahrzeug
  innen / Außenantenne). Der Bildschirm bleibt an (Wake Lock); offline funktioniert die App nach dem
  ersten Aufruf.
* **Export als CSV oder JSON** und im Simulator in den Reiter *Kalibrierung* ziehen. Die
  Kalibrierung wertet dabei aus:
  * den Aufbau pro Zeile (4 / 8 / 0 dB Zusatzverlust)
  * die GPS-Genauigkeit (Zeilen über 30 m werden verworfen)
  * die Repeater-Hashes (Mehrdeutiges geht an den nächstgelegenen Repeater)

**Voraussetzungen**

* Android-Tablet mit Chrome oder Edge: alles geht (Bluetooth, GPS, Installation als App).
* iPad: Safari kann kein Web Bluetooth; mit dem Browser „Bluefy“ geht es, sonst nur manuell.
* GPS und Bluetooth gehen nur über **HTTPS** (oder `localhost`). Für das Tablet muss die App also per
  HTTPS erreichbar sein, z. B. als statische Seite auf funkmesh.de oder GitHub Pages: den Ordner
  `funkmesh_sim/mapper/` 1:1 hochladen. Ein Aufruf über `http://<PC-IP>:8765/mapper/` reicht nicht.

## Datengrundlage

| Daten | Quelle | Lizenz |
|---|---|---|
| Gelände 1 m (DGM1), nur NRW | OpenGeodata.NRW, Laserscanning | dl-de/zero-2.0 |
| Oberfläche 1 m (DOM1) → Bewuchshöhe, nur NRW | OpenGeodata.NRW | dl-de/zero-2.0 |
| 3D-Gebäude LoD2 (CityGML, Dachflächen), nur NRW | OpenGeodata.NRW | dl-de/zero-2.0 |
| Oberflächenmodell 30 m, ganz DE | Copernicus DEM GLO-30 (© DLR e.V. 2010–2014, © Airbus Defence and Space GmbH 2014–2018, bereitgestellt unter COPERNICUS durch EU und ESA) | frei mit Namensnennung |
| Landbedeckung 10 m (Wald, Bebauung), ganz DE | ESA WorldCover 2021 v200 (© ESA WorldCover project / Contains modified Copernicus Sentinel data (2021)) | CC BY 4.0 |
| Gebäudegrundrisse und -höhen außerhalb NRW | © OpenStreetMap-Mitwirkende (Overpass-API) | ODbL |
| Gebiets-/Kreisgrenzen, Ortssuche | © OpenStreetMap-Mitwirkende (Nominatim) | ODbL |
| Repeater-Standorte | map.meshcore.io, mapme.sh (öffentliche APIs) | – |
| Hintergrundkarten | basemap.de/BKG, OSM, DOP NRW | jeweilige Lizenz |

Welche Quelle genutzt wird, entscheidet das Gebiet:

* Raster ≤ 10 m und Kacheln in NRW (bis 12 000 km²) → NRW-1-m-Daten. Am NRW-Rand wird mit den
  deutschlandweiten Daten ergänzt.
* sonst Copernicus + WorldCover; OSM-Einzelgebäude bei Raster ≤ 10 m und Fläche ≤ 4000 km².
* größere oder gröbere Gebiete (Bundesland, Deutschland): Bebauung als 8 m hoher **Clutter** nach dem
  Ansatz von ITU-R P.1812 (Beugung über die Bebauung, keine Innenraumwerte).

Die Copernicus- und WorldCover-Daten werden blockweise direkt aus den Cloud-Optimized-GeoTIFFs gelesen
(HTTP-Range, passende Übersichtsstufe) – es werden keine ganzen Kacheln heruntergeladen.

**Aufbereitung NRW** auf ein 5-m-Raster:

* **Gelände**: Mittelwert des DGM1.
* **Gebäude**: höchste LoD2-Dachfläche je Zelle, Dachebenen exakt gerastert, sofern ≥ 50 % der Zelle
  bebaut sind.
* **Bewuchs**: 75 %-Perzentil von DOM1 − DGM1 über Pixeln > 2 m außerhalb von Gebäuden, sofern
  ≥ 30 % der Zelle bewachsen sind.

Gegenprobe Kachel Coesfeld: Das LoD2-Dach liegt im Median 0,0 m neben dem Laser-DOM (90 % zwischen
−0,5 und +1,4 m).

**Aufbereitung deutschlandweit** (`funkmesh_sim/data/generic.py`), je Block mit 2 km Überlappung:

* **Gelände**: GLO-30 ist ein Radar-*Oberflächen*modell. Wo frei (kein Wald, keine Bebauung laut
  WorldCover/OSM) wird es übernommen, unter Wald und Bebauung aus der Umgebung mehrskalig interpoliert.
  Danach entfernt eine morphologische Öffnung (150 m) Hecken, Baumreihen, Feldgehölze und
  Radar-Rauschen, die sonst für niedrige Antennen wie Beugungskanten wirken; zuletzt 20 m Glättung.
* **Bewuchs**: Waldzellen (Baumanteil ≥ 30 %), Kronenhöhe = 0,6 · (Oberfläche − Gelände) + 9,2 m,
  6–35 m. Das Radar dringt in die Krone ein; die Formel ist an 608 000 Waldzellen des NRW-Laser-DOM
  kalibriert (ohne Korrektur wären die Bäume ~40 % zu niedrig).
* **Gebäude**: OSM-Grundrisse im 2-m-Subraster, Zelle = Gebäude bei ≥ 50 % Überdeckung. Höhe aus
  `height`, sonst `building:levels` × 3 m + Dachgeschosse × 2,5 m + 1,5 m, Garagen/Schuppen 3 m,
  sonst Vorgabe **9 m** (einstellbar; Median der LoD2-Gebäude im Münsterland: 8,3 m).

**Gegenprobe** im Kreis Coesfeld (Dülmen, 10 m Raster): dieselbe Fläche einmal mit NRW-Laserdaten und
einmal mit den deutschlandweiten Quellen aufbereitet:

| Größe | Abweichung deutschlandweit − NRW |
|---|---|
| Gelände (alle Zellen) | 0,0 ± 1,9 m |
| Wald erkannt | 89 % Übereinstimmung |
| Pfadverlust, Repeater 30 m (Kirchturm) | −3,6 dB im Mittel, ± 5,1 dB je Zelle |
| Pfadverlust, Repeater 12 m (Feld) | +1,4 dB, ± 5,7 dB |
| Pfadverlust, Repeater 10 m (Dach, Ort) | +2,0 dB, ± 5,3 dB |

Außerhalb NRW ist das Modell also im Mittel auf wenige dB genau, im Einzelfall (einzelne Straße,
Hecke) deutlich ungenauer als mit Laserdaten. Mit eigenen Messungen (Kalibrierung) lässt sich der
mittlere Fehler je Gebiet beseitigen.

## Ausbreitungsmodell

Für jede Zielzelle wird das Profil Repeater → Empfänger im 5-m-Raster abgetastet (GPU: ein Thread je
Zelle, float32).

1. **Beugung: Delta-Bullington nach ITU-R P.1812-7 / P.526-15.**
   * Die Hindernisse kommen direkt aus dem Profil, in drei Varianten:
     * Gelände
     * Gelände + LoD2-Gebäude
     * Gelände + Gebäude + Bewuchs
   * Dazu kommt die sphärische Erdbeugung `Ldsph` (ITU-R P.452-17 §4.2.2). Effektive Antennenhöhen
     stammen aus der Ausgleichsgeraden des Geländes; die Bullington-Rechnung über glatter Erde wird
     abgezogen.
   * Der Korrekturterm `(1 − e^(−L/6))(10 + 0,02 d)` wird nur für Geländehindernisse angewendet;
     Gebäude und Kronen gelten als scharfe Kanten.
   * Alternativ einstellbar: Deygout mit 3 Kanten.
   * Prüfung über flachem Gelände: Das Ergebnis entspricht genau der sphärischen Erdbeugung (+16 dB
     über Freiraum bei 10 km, Antennen 30 m und 1,5 m).
2. **Bewuchs: ITU-R P.833** `A = Am·(1 − e^(−γ·d/Am))` mit `Am = 0,18·f^0,752` (≈ 29 dB) und
   γ = 0,25 dB/m (belaubt). Gerechnet wird das Minimum aus „über die Krone“ (Beugung) und „durch die
   Krone“.
3. **Gebäudedurchdringung** des direkten Strahls: je Außenwand 10 dB (ITU-R P.2040, Mauerwerk ca.
   30 cm) plus 0,6 dB/m innen. Das Ergebnis wird leistungsmäßig mit dem Pfad über das Dach addiert.
4. **Innenraum**: Gebäudeeintrittsdämpfung nach ITU-R P.2109 (traditionelle Bauweise, Median
   ≈ 14 dB).
5. **Straßenschluchten: ITU-R P.1411** (UHF, NLoS um eine Ecke, Reflexions- und Beugungspfad).
   * Gilt nur, wenn die Repeaterantenne unter dem umgebenden Dachniveau liegt; Grundlage ist das
     Anwendungsgebiet von P.1411.
   * Die Ausbreitung entlang der Straßen läuft über eine Relaxation auf dem Raster, mit voller
     Sichtprüfung Ecke → Zelle.
6. **Antenne**: Vertikaldiagramm einer Kollinear-Antenne passend zum Gewinn, mit Nullstellen steil
   unter dem Mast.
7. **Statistik**:
   * Lokaler Mittelwert über 3×3 Zellen, gemittelt wird die Leistung.
   * Ortsvariabilität σ = 5,5 dB außen; innen zusätzlich 8 dB für den Gebäudeeintritt.
   * Daraus folgen Versorgungswahrscheinlichkeit und Redundanz.
8. **Linkbudget**:
   * Downlink und Uplink werden getrennt gerechnet.
   * Endgerät: 22 dBm, 0 dBi, 4 dB Körperdämpfung, 1,5 m über Grund.
   * LoRa-Empfindlichkeit = −174 + 10·log10(BW) + NF + SNRmin(SF). EU Narrow (SF8/62,5 kHz):
     −130 dBm.

**Auslastung** (`funkmesh_sim/sim/traffic.py`): Die ereignisgesteuerte Simulation bildet
MeshCore-Flooding nach:

* Duplikaterkennung; der Pfad wächst um 1 Byte je Hop.
* Wiederholverzögerung `U(0,5)·txdelay·Airtime`.
* Listen-before-Talk, Airtime-Faktor und Halbduplex.
* Kollisionen mit 6 dB Capture-Schwelle.
* Endgeräte werden zufällig auf Gebäude im Kreis verteilt.

Ausgegeben werden Duty Cycle je Repeater (EU-Grenze 10 % im Band 869,4–869,65 MHz), Kanalbelegung,
Empfangsverluste und Flood-Reichweite.

### Kalibrierung an Messungen

Reiter **Kalibrierung**: Messdateien in das Feld ziehen (oder auswählen). Angenommen werden der CSV-
und JSON-Export der Mapper-App sowie beliebige CSV-Dateien mit einer Zeile je Paket: `lat`, `lon`,
`rssi`, `repeater`, optional `snr`, `direction` (`down`/`up`), `height`, `setup`, `gps_accuracy`.
Spaltennamen deutsch oder englisch, Trennzeichen `,` oder `;`.

* Jede Datei wird ein **Datensatz** des Gebiets (einzeln abschaltbar oder löschbar). Ausgewertet
  werden immer alle aktiven Datensätze zusammen.
* Mit „automatisch auswerten und übernehmen“ (Standard) wird nach jedem Hinzufügen neu gefittet und
  das Ergebnis (Offset, γ, Beugungsmodell, σ) **übernommen, wenn es plausibel ist** (Reststreuung
  ≤ 12 dB). Danach „Ausleuchtung neu berechnen“.
* Probe mit synthetischen Messungen (Modell + 3 dB + 6 dB Rauschen, 260 Punkte): erkannt wurden
  2,6 dB Offset und σ = 5,9 dB.

* `repeater` darf ein Name, die Projekt-ID oder ein Public-Key-Präfix sein, z. B. der 1-Byte-Hash
  aus einem MeshCore-Trace. Mehrdeutige Präfixe gehen an den nächstgelegenen Repeater.
* Ein leeres `rssi` bedeutet „nicht empfangen“ und fließt in die Trefferquote ein.
* SNR < 0: Der gemeldete RSSI enthält Rauschen, deshalb gilt Nutzsignal = RSSI + SNR (Semtech).
* GPS-Punkte in Gebäudezellen werden auf die nächste freie Zelle (≤ 10 m) verschoben.
* **Zensierung**: Nahe der Empfindlichkeitsgrenze werden nur zufällig starke Pakete empfangen.
  Deshalb zählen für Mittelfehler und Fit nur Punkte, deren Vorhersage ≥ 5 dB über der
  Empfindlichkeit liegt.
* Auswertung:
  * Mittelfehler, Streuung, RMSE
  * Trefferquote (Empfang vorhergesagt vs. gemessen)
  * Fehler nach Entfernung und je Repeater (großer Fehler bei einem Repeater = meist falsche Höhe
    oder Leistung)
  * Streudiagramm und Messpunkte farbig auf der Karte
* **Fit**: Gitter über Bewuchsdämpfung γ (0,05–0,6 dB/m) × Beugungsmodell. Daraus folgen ein
  Vorschlag für den Pegel-Offset und σ, per Knopf übernehmbar.
* Die Profilstreifen werden einmal extrahiert und dann je Variante parallel ausgewertet: rund 400
  Messungen × 18 Varianten in wenigen Sekunden.
* Der Offset (`stats.calib_offset_db`) wirkt auf alle Budgets, ohne dass der Cache neu gerechnet
  werden muss.

**mapme.sh als Quelle** (Knopf im Kalibrier-Reiter): lädt für jeden Repeater des Gebiets mit Public
Key die Community-Messungen je H3-Wabe (`/api/node/<key>/coverage`) und legt sie als Datensatz an
(Messaufbau „Fahrzeug, Gerät innen“).

* Verworfen werden Waben über 40 km Entfernung sowie Waben, bei denen ein anderer bekannter Repeater
  mit gleichem 1-Byte-Hash weniger als doppelt so weit entfernt ist.
* Gewichtet wird nach Samples, gedeckelt auf 20.
* Erster Abgleich Coesfeld (Stand 22.11.2026), 876 Waben von 22 Repeatern:
  * Streuung ~29 dB, nicht plausibel.
  * Nah am Repeater wird schwächer gemessen als vorhergesagt, in 5–40 km Entfernung stärker.
  * Mögliche Gründe: falsche Repeaterhöhen (map.meshcore.io und mapme.sh liefern beide nur 10 m als
    Standard) und Mapper auf Hügeln oder Türmen (Spielmechanik von mapme.sh). Die Mittelung des
    RSSI je Wabe ist nicht dokumentiert.
* Die Daten eignen sich daher vor allem, um Repeater mit falschen Angaben zu finden (Tabelle
  „Nach Repeater“) und für die Trefferquote. Ein globaler Fit wird bei einer Reststreuung über 12 dB
  gesperrt.

Messaufbau beachten: zu Fuß am Körper ca. 4 dB Zusatzverlust, im Fahrzeug ohne Außenantenne ca. 8 dB,
Stativ oder Dachantenne 0 dB.

### Grenzen

Die Ergebnisse sind Modellwerte ohne Kalibrierung an Messungen. Außerhalb NRW sind die Geodaten
deutlich gröber (siehe Gegenprobe oben); bei Gebieten mit 25–100 m Raster ist das Ergebnis ein
Überblick, keine Straßenplanung. Nicht enthalten sind:

* Fahrzeuge und andere bewegliche Hindernisse
* beschichtete Fenster
* die Antennenumgebung direkt am Mast
* Fremdstörungen

Die Antennenhöhen der importierten Repeater sind auf 10 m gesetzt und **müssen gepflegt werden**,
weil weder map.meshcore.io noch mapme.sh echte Höhen liefern.

Empfehlung: Messfahrten bzw. -gänge machen und über den Reiter „Kalibrierung“ einspielen.

## Entwicklung

```bash
.venv/bin/pip install -e ".[gpu,dev]"
.venv/bin/python -m pytest
```

Struktur:

* `areas.py`: Gebiete (Planung, Abschätzung, Aufbereitung, Ortssuche)
* `data/`: Datenpipeline (`nrw.py` + `lod2.py` für NRW, `generic.py` deutschlandweit)
* `radio/kernel.py`: Ausbreitungskernel, ein Quelltext für CUDA und CPU
* `radio/canyon.py`: Straßenschlucht-Nachlauf
* `radio/engine.py`: Fenster, Speicher, Cache
* `coverage.py`: Kombination und Links
* `sim/`: Auslastungssimulation
* `server/`: API und Kacheln
* `static/`: Oberfläche

## Lizenz

MIT. Geodaten unterliegen ihren jeweiligen Lizenzen (siehe oben).
