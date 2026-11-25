# Zwischen zwei Verteilungen – verschieben statt mischen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-profile-verschieben-demo.streamlit.app/)**

Demo der Reihe **optimaler-transport** („Optimaler Transport: Verteilungen ineinander verschieben“) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Sie setzt an zwei Annahmen an, die die Demos „Vom Zuordnen zum Transportieren“ und „Abstand zwischen Verteilungen“ stehen lassen: *Bei Gleichstand ist die Wahl des optimalen Plans für den Wert gleichgültig.* und die Frage, wie man von einer Verteilung zur anderen kommt (die Demo „Abstand zwischen Verteilungen“ misst nur den Abstand). Drei Experimente prüfen die Annahme „Die Mitte zweier Verteilungen ist ihre Mischung“: Zwischenschritte (Experiment 1), Tagesprofile dreier Standorte (Experiment 2) und unterschiedliche Morgenanteile (Experiment 3).

## Warum dieses Problem

„Die Mitte“ oder „ein Schritt zwischen zwei Verteilungen“ steht in vielen Sätzen: beim Mitteln von Prognosen, beim Vergleich von Nachfrageprofilen mehrerer Standorte, beim Übergang von einem Profil zum anderen. Es gibt zwei Wege, und sie beantworten verschiedene Fragen: Die **Mischung** (englisch mixture) mittelt die Wahrscheinlichkeiten punktweise und ist die Gerade der Totalvariation; die **Verschiebung** (englisch displacement interpolation) mittelt die Quantile, jede Masse legt den Bruchteil $t$ ihres Weges im optimalen Plan zurück, und sie ist die Gerade der W2-Geometrie. Die Demo zeigt, wann die Mischung die richtige Antwort ist (welcher Zustand tritt ein? wie hoch ist die Nachfrage aller Standorte zusammen?) und wann nicht (wie sieht ein Profil mit mittlerer Phase aus?), und dass auch das **Baryzentrum** (Wasserstein-Baryzentrum, bei mehr als zwei Verteilungen die Verschiebung) nicht allgemein die Gestalt erhält. Vehikel sind Normalverteilungen und **Tagesprofile**: die Verteilung der Nachfrage über 24 Stunden (Objekt Zeit). „Tagesprofil“ heißt hier immer eine Verteilung über 24 Stunden; ein einzelner Tageswert ist etwas anderes.

## Modell

- **Verteilungen:** Mischungen von Normalverteilungen und Atome (Punktmassen) auf der Zeitachse, erzeugt, keine Daten. Das **Tagesprofil** hat Spitzen bei 8 und 18 Uhr mit den Anteilen 0,4 und 0,6 und der Streuung 1,2 Stunden je Spitze. **Drei Standorte** haben gleiches Gewicht $\tfrac13$ und sind in Experiment 2 gegeneinander um $-s$, $0$, $+s$ Stunden verschoben, in Experiment 3 haben sie dieselben Spitzen und die Morgenanteile $0{,}4 - \delta$, $0{,}4$, $0{,}4 + \delta$.
- **Kosten:** quadrierte Stunden (W2); das Wort „Wasserstein“ steht für sich allein nur im Namen „Wasserstein-Baryzentrum“ (gemeint ist immer W2); sonst steht es mit seiner Ordnung (W1 oder W2). Die Uhrzeit ist ein Kreis; hier wird sie als **Linie** behandelt, die Masse außerhalb von 0 bis 24 Uhr steht in der Tabelle von Experiment 2.
- **Mischung und Verschiebung:** $P_t^{\text{Mischung}} = (1 - t)\,P + t\,Q$ und $Q_t^{\text{Verschiebung}}(u) = (1 - t)\,Q_P(u) + t\,Q_Q(u)$ mit den **Quantilfunktionen** (englisch quantile functions) $Q_P$, $Q_Q$. In 1D ist $W_2^2(P, Q) = \int_0^1 (Q_P(u) - Q_Q(u))^2\,\mathrm{d}u$. Das **Baryzentrum** ist der Minimierer von $\sum_k \lambda_k W_2^2(\nu, \nu_k)$; in 1D ist es für nichtatomare Maße das Mittel der Quantilfunktionen (Agueh, Carlier 2011, Abschnitt 6.1, im Wortlaut gelesen); für Punktmassen (Atom-Fassung) steht das dort nicht, die Demo prüft es selbst gegen ein Linearprogramm.
- **Gipfel** (englisch mode) sind lokale Maxima der Dichte mit einer **Prominenz** (englisch prominence) von mindestens einem Anteil der höchsten Dichte, gemessen über dem höheren der beiden Täler zu den nächsten höheren Gipfeln (Definition von `scipy.signal.find_peaks`); Standard 2 %.
- **Spitzenstunde** (englisch peak hour): größter Anteil der Tagesnachfrage in einem Intervall von einer Stunde; **Fensterdeckung**: Anteil der Gesamtnachfrage in den zwei Fenstern 8 ± 2,4 und 18 ± 2,4 Uhr (zwei Streuungen).
- Formeln und Handbeispiele stehen im Expander „📐 Mathematische Formulierung“ der App. Eine ausführliche, verständlich geschriebene Erklärung mit Fachteil steht auf der Seite [Verschieben statt mischen verständlich erklärt](https://sebastianhanisch.net/verschieben-statt-mischen-erklaert.html); die App fasst sie im Expander „💬 In einfachen Worten“ zusammen.

## Methodik

- **Rechnung:** eindimensional, deterministisch, ohne Zufall und ohne Zufalls-Seed. Verteilungsfunktionen und Dichten auf einem Gitter von 0,001 Stunden (die Gipfelzählung der Tagesprofile von −6 bis 30 Uhr, die Quantile bis ±9 Streuungen um die Komponenten), Quantile durch lineare Umkehrung der Verteilungsfunktion, Integrale über 20 000 Quantilstufen in der Normalskala (Mittelpunktsregel). Die Dichte des Baryzentrums ist ohne Glättung das gewichtete harmonische Mittel der Dichten an den gleichstufigen Quantilen: $Q'(u) = \sum_k \lambda_k / f_k(Q_k(u))$.
- **Atom-Fassung:** Experiment 3 rechnet zusätzlich mit Punktmassen (jede Spitze eine Punktmasse bei 8 bzw. 18 Uhr) exakt in Brüchen (`fractions`).
- **Zählweise:** Gipfel (Prominenz wählbar in Experiment 2, sonst 2 %), neun Zwischenschritte $t = 0{,}1$ bis $0{,}9$, Anteile der Masse in Fenstern.
- **Module:** `pvs_verteilung.py` (Mischungen, Quantile, Dichte des Quantilmittels, Gipfelzählung), `pvs_diskret.py` (Brüche), `pvs_profile.py` (Tagesprofil, Standorte, Kennzahlen), `pvs_evaluation.py` (Reihen und Verdikte), `pvs_normal.py` (Normalverteilung ohne scipy, wörtliche Kopie aus „Mitte in anderen Räumen: Kreis und Verteilungen“).
- **Zufall:** keiner in der App; `pvs_rng.py` (SplitMix64, bytegleich zur Vorlage des Portfolios) dient nur den Eigenschaftstests.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stammen aus der Rechnung der Demo (Normalverteilungen und Atome, 1D, Quadratkosten; erzeugte Profile, keine Realdaten). Gezählt wird in den neun Zwischenschritten $t = 0{,}1$ bis $0{,}9$ („k von 9“); Gipfel bei einer Prominenz von mindestens 2 % der höchsten Dichte. „Streuung ok“ heißt: höchstens 10 % neben der Vorgabe $(1 - t) s_0 + t s_1$.

### Experiment 1: Verschieben oder mischen

| Enden | D (Stunden) | Mischung eingipfelig | Mischung: Streuung ok | Verschiebung eingipfelig | Verschiebung: Streuung ok | W2-Verhältnis der Mischung bei t = ½ |
|---|---|---|---|---|---|---|
| N(0, 1) → N(D, 1) | 1 | 9 von 9 | 4 von 9 | 9 von 9 | 9 von 9 | 1,028 |
| N(0, 1) → N(D, 1) | 2 | 9 von 9 | 0 von 9 | 9 von 9 | 9 von 9 | 1,085 |
| N(0, 1) → N(D, 1) | 3 | 4 von 9 | 0 von 9 | 9 von 9 | 9 von 9 | 1,142 |
| N(0, 1) → N(D, 1) | 4 | 0 von 9 | 0 von 9 | 9 von 9 | 9 von 9 | 1,188 |
| N(0, 1) → N(D, 1) | 6 | 0 von 9 | 0 von 9 | 9 von 9 | 9 von 9 | 1,250 |
| N(0, 1) → N(D, 1) | 8 | 0 von 9 | 0 von 9 | 9 von 9 | 9 von 9 | 1,286 |
| N(0, 1) → N(D, 3²) | 0 | 9 von 9 | 4 von 9 | 9 von 9 | 9 von 9 | 1,264 |
| N(0, 1) → N(D, 3²) | 4 | 7 von 9 | 0 von 9 | 9 von 9 | 9 von 9 | 1,283 |

Für Normalverteilungen ist die Verschiebung eingipfelig und die Streuung die Vorgabe **per Definition** (das Quantilmittel zweier Normalverteilungen ist eine Normalverteilung mit linear wechselnder Lage und Streuung); die Spalten der Verschiebung sind kein Befund, die der Mischung sind es. Das W2-Verhältnis der Mischung wächst mit $D$; im Grenzfall getrennter Träger (Punktmassen) kostet die Mischung $W_2^2 = t D^2$ gegen $(t D)^2$ der Verschiebung, das Verhältnis der W2 ist $1/\sqrt{t}$, bei $t = \tfrac12$ also $\sqrt 2 = 1{,}414$ (exakt, Handbeispiel); die gemessenen Werte liegen darunter; bei gleicher Streuung nähert sich das Verhältnis mit wachsendem $D$ dem Grenzwert, weil sich die Normalverteilungen weniger überlappen (D = 24: 1,369, außerhalb des Reglers). In der Zeile N(0, 1) → N(0, 3²) bei D = 0 gibt es keinen Abstand der Lagen; dort kommt der Wert von der Streuungsänderung.

Glocke N(0, 1) → zwei Blöcke bei −4 und +4 (Streuung 0,5), Gipfelzahl von t = 0,1 bis 0,9: Mischung 3, 3, 3, 3, 3, 3, 3, 3, 3; Verschiebung 2, 2, 2, 2, 2, 2, 2, 2, 2; das W2-Verhältnis der Mischung bei t = ½ ist 1,237.

Ein Tagesprofil mit Spitzen bei 8 und 18 Uhr (Streuung 1,2 Stunden) und dasselbe um $D$ Stunden versetzt:

| Versatz D (Stunden) | Gipfel der Mischung, t = 0,1 bis 0,9 | Gipfel der Verschiebung | W2-Verhältnis der Mischung bei t = ½ |
|---|---|---|---|
| 2 | 2, 2, 2, 2, 2, 2, 2, 2, 2 | 2, 2, 2, 2, 2, 2, 2, 2, 2 | 1,065 |
| 4 | 2, 4, 4, 4, 4, 4, 4, 4, 2 | 2, 2, 2, 2, 2, 2, 2, 2, 2 | 1,158 |

Für N(0, 1) → N(4, 1) liegt die Mischung in der Totalvariation auf der Geraden (TV(A, Mischung_t) = t · TV(A, B), über die 21 Werte von t in Schritten von 0,05 bis auf Rundung, Abweichung unter 1e-12) und die Verschiebung in W2 (W2(A, Verschiebung_t) = t · W2(A, B), ebenso); umgekehrt liegt die Verschiebung bei t = ½ in der Totalvariation bei 0,715 von TV(A, B) statt bei 0,500, und die Mischung in W2 beim 1,188-fachen des halben Weges.

### Experiment 2: Tagesprofile, Summe oder typisches Profil

Drei Standorte mit demselben Profil bei $-s$, $0$, $+s$ Stunden, gleiche Gewichte, Normalverteilungen mit $\sigma = 1{,}2$ Stunden, Prominenz 2 %:

| s (Stunden) | Gipfel der Mischung | Streuung der Mischung (Stunden) | Fensterdeckung der Summe | Spitzenstunde: Baryzentrum zu Summe | Masse der Summe außerhalb 0 bis 24 Uhr |
|---|---|---|---|---|---|
| 0,5 | 2 | 5,060 | 94,18 % | 1,06 | 0,000 % |
| 1 | 2 | 5,109 | 90,22 % | 1,23 | 0,000 % |
| 2 | 2 | 5,302 | 73,85 % | 1,95 | 0,009 % |
| 3 | 6 | 5,607 | 52,39 % | 2,72 | 0,124 % |
| 4 | 5 | 6,009 | 37,94 % | 2,42 | 0,962 % |
| Baryzentrum (jedes s) | 2 | 5,044 | 95,45 % | – | – |

Streuung der Mischung (Stunden), geschlossen aus dem Gesetz der totalen Varianz $\sqrt{25{,}44 + 2 s^2 / 3}$: s = 1: 5,109; s = 2: 5,302; s = 3: 5,607; s = 4: 6,009.

Das Baryzentrum von Verschiebungen mit Mittel 0 ist das mittlere Profil selbst: ein Satz für Verschiebungsfamilien (die Demo rechnet es über die Quantile nach, größter Abstand der Quantilfunktionen zum Profil unter 1e-9 Stunden für alle 17 Werte von $s$ des Reglers, alle Vielfache des Gitterschritts), keine Messung; deshalb hat es immer die zwei Gipfel des Profils. „Mittlere Phase“ heißt hier die Mitte der Versätze $-s$, $0$, $+s$ im W2-Sinn (Mittel der Quantile), auf der Linie. Die Sechs der Mischung ist eine **Zählung** und gilt nur in einem Fenster von $s$ und für ein Prominenz-Kriterium. Gipfel der Mischung bei Prominenz 2 % für $s$ von 0 bis 8 Stunden in Schritten von 0,25 (der Regler der App reicht bis 4): 2 (s = 0 bis 2,5), 6 (s = 2,75 bis 3,25), 5 (s = 3,5 bis 6,5), 6 (s = 6,75), 4 (s = 7 bis 8). Mit nur **zwei** um $s$ versetzten Profilen hat die Mischung (Prominenz 2 %, $s$ von 0 bis 8 in Schritten von 0,25) höchstens vier Gipfel; die Sechs braucht drei Standorte.

Gipfel der Mischung bei s = 3 Stunden je Prominenz-Kriterium: 0,5 %: 6; 1,0 %: 6; 2,0 %: 6; 5,0 %: 6; 10,0 %: 4; bei s = 2,75 Stunden sind es 6 bei 2 % und 2 bei 5 %.

Für die **Summe der Standorte** ist die Mischung (Summe geteilt durch drei) die richtige Antwort: bei $s = 3$ decken die aus dem Baryzentrum abgelesenen Fenster nur 52,39 % der Gesamtnachfrage statt 95,45 %, und die Spitzenstunde des Baryzentrums ist das 2,72-Fache (Tabelle). In der Atom-Fassung (jede Spitze eine Punktmasse bei 8 bzw. 18 Uhr, s = 3) ist die höchste Einzelmasse des Baryzentrums das 3-Fache der Mischung (hier gleich der Zahl der Standorte, weil die sechs Orte der Mischung verschieden sind). Die Fensterdeckung ist eine andere Größe als der Servicegrad in „Mitte in anderen Räumen: Kreis und Verteilungen“ (dort Lieferanten, nicht Nachfrageprofile); die Zahlen sind nicht dieselben.

### Experiment 3: unterschiedliche Morgenanteile

Drei Standorte mit denselben Spitzen und den Morgenanteilen $0{,}4 - \delta$, $0{,}4$, $0{,}4 + \delta$; Masse zwischen den Spitzen (10,4 bis 15,6 Uhr):

| δ | Atome: Masse des Baryzentrums zwischen den Spitzen (exakt) | Normalverteilungen: Baryzentrum | Normalverteilungen: Mischung | Normalverteilungen: jeder Standort (höchstens) |
|---|---|---|---|---|
| 0 | 0 | 2,28 % | 2,28 % | 2,28 % |
| 0,05 | 1/10 | 10,02 % | 2,28 % | 2,28 % |
| 0,1 | 1/5 | 19,95 % | 2,28 % | 2,28 % |
| 0,2 | 2/5 | 38,29 % | 2,28 % | 2,28 % |

In der Atom-Fassung liegt exakt der Bruchteil $2\delta$ der Masse des Baryzentrums zwischen den Spitzen, **wo kein Standort Nachfrage hat**; bei Normalverteilungen hat jeder Standort dort 2,28 %, und die Mischung bleibt bei diesem Wert (die Aussage „dort hat kein Standort Nachfrage“ gilt nur für die Atom-Fassung).

Bei δ = 0,1 (Morgenanteile 0,3 / 0,4 / 0,5): Atome: das Baryzentrum hat die Orte 8 (3/10), 34/3 (1/10), 44/3 (1/10), 18 (1/2), die Mischung 8 (2/5), 18 (3/5); Mittel der quadrierten Verschiebungen 40/9 gegen 20/3; Varianzen der Standorte 21, 24, 25, des Baryzentrums 176/9, der Mischung 24. Normalverteilungen (σ = 1,2 Stunden): 4 Gipfel des Baryzentrums gegen 2 der Mischung, Standardabweichung 4,812 gegen 5,044 Stunden, Mittel der quadrierten Verschiebungen 2,289 gegen 3,026 Stunden².

Das Baryzentrum ist schmaler als die Mischung, auch wenn es Masse zwischen die Spitzen legt (als Ungleichung gilt: Standardabweichung des Baryzentrums ≤ mittlere Standardabweichung der Standorte ≤ Standardabweichung der Mischung; geprüft an 40 zufälligen Verteilungen aus Punktmassen und an den Normalverteilungen dieser Demo); es minimiert den Mittelwert der quadrierten Verschiebungen zu den Standorten, nicht die Treue zur Gestalt.

### Gleichstand: zwei gleich teure Zuordnungen, verschiedene Zwischenverteilungen

Zwei Verteilungen mit je zwei gleich schweren Orten, $P$ bei 0 und 1, $Q$ bei 2 und 3:

| Zuordnung | Kosten \|x − y\| (Mittel) | Kosten (x − y)² (Mittel) | Zwischenverteilung bei t = ½: Orte (Masse) |
|---|---|---|---|
| 0 → 2, 1 → 3 | 2 (optimal) | 4 (optimal) | 1 (1/2), 2 (1/2) |
| 0 → 3, 1 → 2 | 2 (optimal) | 5 | 3/2 (1) |

Bei den Kosten $|x - y|$ sind beide Zuordnungen optimal, der Wert ist derselbe (2), die Zwischenverteilungen sind verschieden; bei $(x - y)^2$ ist in 1D die sortierte Zuordnung die einzige optimale, und sie liefert das Quantilmittel.

### Zahl → Test

| Zahl oder Satz | Test (Band oder exakt) |
|---|---|
| alle Tabellen und Messsätze dieser README | `tests/test_claims.py::test_stuecke_der_readme_stehen_wortwoertlich_in_der_readme` (aus dem Code gebildet, gleiche Funktionen wie die App) |
| Handbeispiele 1 bis 5, 7, 8 in Brüchen (W2² 4, 1, 2; Orte 9 und 19; 34/3 und 44/3; 1/5; 40/9 gegen 20/3; Varianzen 21, 24, 25) | `tests/test_diskret.py` (exakt) |
| 33 Werte der Gipfelzahl, Prominenz 0,5 bis 10 % | `tests/test_verteilung.py` (Zählungen) |
| Streuung 5,044, 5,607, 6,009; Fensterdeckung 95,45 % und 52,39 %; Spitzenstunde 2,72; Masse außerhalb 0,12 % und 0,96 % | `tests/test_profile.py` (Band 5e-4 gegen die Handformel) |
| Quantilmittel gegen Linearprogramm (40/9), gegen Aufzählung, gegen die geschlossene Form der Normalverteilung, Gipfel gegen `scipy.signal.find_peaks` | `tests/test_oracle_profile_verschieben.py` |
| Kantensätze, Konstanten, Testzahl | `tests/test_claims.py` |

## Ehrliche Grenzen

Die Annahmen der Vorgänger und was die Demo daran ändert:

- *Bei Gleichstand ist die Wahl des optimalen Plans für den Wert gleichgültig.* Für den Wert ja, für die Zwischenverteilung nicht (Kosten $|x - y|$; Tabelle oben).
- *Kontrast, keine behobene Schwäche:* Die Demo „Abstand zwischen Verteilungen“ misst den Abstand zwischen zwei Verteilungen. Diese Demo stellt die zwei Geodäten gegenüber: die *Mischung* (Gerade der Totalvariation) gegen die *Verschiebung* (Gerade der W2-Geometrie).

Die Annahme, die die Demo prüft, und ihre Grenzen:

- *Die Mitte zweier Verteilungen ist ihre Mischung.* Das gilt für die Frage, welcher Zustand eintritt, und für die Summe der Standorte; nicht für die Gestalt eines Profils mit mittlerer Phase.
- *Das Baryzentrum ist das mittlere Profil.* Das gilt nur für Verschiebungsfamilien (gleiche Anteile, nur die Phase ändert sich); bei verschiedenen Anteilen legt es Masse zwischen die Spitzen (Experiment 3).

Weitere Grenzen:

- Alle Aussagen gelten in diesem Modell, bei diesen Parametern und mit der genannten Zähleinheit: eindimensional, Quadratkosten, Normalverteilungen oder Atome, drei Standorte gleichen Gewichts, erzeugte Profile.
- Die Zahl „sechs Gipfel bei $s = 3$“ gilt für die Prominenz-Schalter 0,5, 1, 2 und 5 %, nicht für 10 %; die Gipfelzahl der Mischung ist nicht monoton in $s$. Das Gitter (0,001 Stunden) wurde nicht variiert.
- Nicht gemessen: zwei und mehr Dimensionen, mehr als zwei Verteilungen im Raum, mehr als drei Standorte, ungleiche Gewichte, echte Lastprofile, der Kreis (Mitternacht): bei $s = 4$ liegen 0,96 % der Masse der Summe außerhalb von 0 bis 24 Uhr.
- Die Gipfelzahl des Baryzentrums in Experiment 3 ist die der exakten Dichte aus den Quantilen, nicht einer geglätteten.
- Literatur: Agueh, Carlier (2011), *Barycenters in the Wasserstein space*, SIAM J. Math. Anal. 43 (Preprint vom 10.12.2010 im Wortlaut gelesen, eine frühere Fassung vom 17.08.2010 ebenfalls; Abschnitt 6.1 gibt das Mittel der Quantilfunktionen für nichtatomare Maße mit endlichem zweiten Moment an, für Punktmassen prüft die Demo es selbst gegen ein Linearprogramm; die Seitenangaben der Zeitschrift sind nicht geprüft); McCanns Interpolation nur nach Agueh, Carlier zitiert, McCann selbst nicht gelesen; Le Coz, Tantet, Flamary, Plougonven (arXiv 2310.17933, Fassung 3 von 2026) ist ein Preprint, nur das Abstract gelesen. Quantilmittel, Interpolation und Baryzentrum sind Lehrstoff oder Forschung; hier wird keine Neuheit behauptet, und es wird nichts über den Einsatz von Baryzentren in der Logistik behauptet.

## Verifikation

- **Handwerte** (`tests/test_diskret.py`, `tests/test_profile.py`, `tests/test_verteilung.py`): die Handbeispiele in Brüchen (zwei Profile 2 Stunden versetzt, drei Standorte, Morgenanteile 3/10, 4/10, 5/10, Gleichstand), Streuungen und Fensterdeckungen aus der Handformel, 33 Werte der Gipfelzahl.
- **Orakel** (`tests/test_oracle_profile_verschieben.py`): Quantilmittel gegen Linearprogramm (HiGHS) auf einem Raster und gegen 100 zufällige Kandidaten, $W_2^2$ in Brüchen gegen die Aufzählung aller Zuordnungen (25 Instanzen), Normalverteilungen gegen die geschlossene Form (W2² = Lage² + Streuung², TV = $2\Phi(D/2) - 1$), Quantile einer Mischung gegen die Verteilungsfunktion aus scipy und gegen Ziehungen aus der Definition, Gipfelzählung gegen `scipy.signal.find_peaks` (60 Zufallsfelder, 8 Werte von $s$ × 5 Kriterien), Normalverteilung und Zufallsgenerator gegen ihre Quellen. Die Datei braucht scipy (`requirements-dev.txt`); fehlt scipy, überspringt sie sich ganz, und ein Test in `tests/test_runtime_imports.py` schlägt dann fehl.
- **Eigenschaften** (`tests/test_eigenschaften.py`, 10 bis 40 Instanzen, SplitMix64): Verschieben aller Standorte verschiebt das Baryzentrum, Vertauschen ändert es nicht, Gewicht 1 gibt das Profil, t = 0 und t = 1 geben die Enden, $W_2$ und TV wachsen linear, dasselbe Mittel, Streuung des Baryzentrums ≤ mittlere Streuung ≤ Streuung der Mischung, Spiegeln.
- **Randfälle** (`tests/test_edge_cases.py`): gleiche Profile, disjunkte Träger, ein Atomort, zwei gleiche Standorte, t am Rand, $s = 0$, Masse außerhalb des Tages, jeder Regler an den Grenzen.
- **Claims** (`tests/test_claims.py`): jede Tabelle und jeder Messsatz der README mit Zahlen wird aus dem Code gebildet und muss wörtlich in der README stehen.
- **Oberfläche:** AppTest (Standard, jedes Schnellstart-Beispiel, Permalink mit Begrenzen und Einrasten, jeder Regler am Minimum und Maximum, alle Maxima zugleich), Footer, Laufzeit-Importe, Layout (Schnellstart oben in der Seitenleiste, fester Kopf, aktives Beispiel hervorgehoben, breite Tabellen scrollen auf dem Telefon).
- **Testzahl:** 251 Tests (`pytest --collect-only -q`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App (Gerüst nach DEMO-PLAYBOOK §2) |
| `pvs_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Schnellstart-Beispiele |
| `pvs_diskret.py` | Punktmassen in Brüchen: Quantilmittel, Mischung, W2², TV, Gleichstand-Beispiel |
| `pvs_evaluation.py` | Reihen der drei Experimente, Kennzahlen, Verdikte (reine Funktionen) |
| `pvs_layout.py` | Fester Kopf (Stylesheet) |
| `pvs_normal.py` | Normalverteilung ohne scipy (wörtliche Kopie) |
| `pvs_presets.py` | Permalink (Begrenzen, Einrasten), Schnellstart in der Seitenleiste, aktives Beispiel hervorgehoben |
| `pvs_profile.py` | Tagesprofil, Standorte, Fensterdeckung, Spitzenstunde, Masse außerhalb |
| `pvs_rng.py` | SplitMix64 (bytegleich zur Vorlage), nur für Tests |
| `pvs_verteilung.py` | Mischungen von Normalverteilungen, Quantile, Dichte des Quantilmittels, Gipfelzählung |
| `pvs_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt: streamlit, numpy, plotly) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (Gegenproben) |
| `tests/` | Handwerte, Orakel, Eigenschaften, Randfälle, Verdikte, Claims, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- Zwei und mehr Dimensionen sowie das Baryzentrum aus mehr als zwei Maßen im Raum: es fehlen ein Verfahren mit Prüfwert und eine Messgröße für „Gipfel“.
- Der Kreis (Mitternacht), mehr als drei Standorte, ungleiche Gewichte und echte Lastprofile.
- Ein Vergleich von Prognosen (das Mitteln von Prognosen mit Servicegrad und CRPS steht in „Mitte in anderen Räumen: Kreis und Verteilungen“).
- PDF-Export und Daten-Eingabe.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Optimaler Transport: vom Plan zum Abstand](https://sebastianhanisch.net/konzepte-optimaler-transport.html).
