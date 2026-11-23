# Ontológiou podporený RAG pre doménu reštaurácií

## 1. Zadanie

| # | Úloha | Riešenie |
|---|---|---|
| 1 | Sprevádzkovať RAG lokálne | Python projekt `restaurant-rag`, spustenie podľa `README.md` |
| 2 | Napojiť RAG na ontológiu | `rag/ontology.py` – ontológia z WebProtégé sa používa pri indexovaní aj pri vyhľadávaní |
| 3 | Pridať ~10 PDF z domény | 12 verejne dostupných PDF (menu reštaurácií, diétne a alergénové príručky, private dining) |
| 4 | Zobraziť graf | `visualize.py` (celá ontológia) + graf použitých konceptov pri každej odpovedi vo webovom rozhraní `server.py` |
| 5 | Dokumentácia + obhajoba | tento dokument, vyhodnotenie v kap. 7 |
| ★ | Rozšírenie nad rámec zadania | **GraphRAG**: znalostný graf (277 jedál) + otázka → SPARQL, kap. 10 |

## 2. Postup RAG (podľa Databricks *RAG workflow*)

```
          INDEXOVANIE (ingest.py)                              DOPYT (ask.py / server.py)
 ┌──────────────┐   ┌──────────────┐               ┌────────────┐
 │ 12 PDF       │   │ restaurant.  │               │  otázka    │
 │ data/pdfs    │   │ owl          │               └─────┬──────┘
 └──────┬───────┘   └──────┬───────┘                     │ 1. nájdenie konceptov ontológie
        │ extrakcia,       │ verbalizácia                │ 2. rozšírenie dopytu (podtriedy)
        │ čistenie,        │ (fakty v texte)             ▼
        │ chunking         │                ┌─────────────────────────────┐
        ▼                  ▼                │ dense (otázka)              │
 ┌──────────────────────────────┐  ───────► │ dense (termíny ontológie)   │──► RRF ──► top-k PDF chunkov
 │ embeddingy + BM25 index      │           │ BM25 (otázka + termíny)     │
 │ index/                       │           └─────────────────────────────┘
 └──────────────────────────────┘                     + fakty ontológie o nájdených konceptoch
                                                              │
                                                              ▼
                                            prompt (kontext + otázka) ──► LLM ──► odpoveď s citáciami [n]
```

1. **Ingest a normalizácia** – `rag/documents.py`: text z PDF (pypdf), odstránenie delení slov a prázdnych riadkov.
2. **Chunking, embedding, index** – chunky ~700 znakov po celých vetách s prekryvom 150 znakov.
   Každý chunk dostane hlavičku s názvom dokumentu (*contextual chunk header*).
   Embeddingy: `paraphrase-multilingual-MiniLM-L12-v2` (viacjazyčný, rovnaký model pre dokumenty aj otázky).
   Popri vektoroch sa stavia BM25 index pre kľúčové slová.
3. **Retrieval** – hybridné vyhľadávanie (vektorové + BM25), zlúčené cez *Reciprocal Rank Fusion* (k = 60).
   Potom **reranking**: top-20 kandidátov prehodnotí cross-encoder `mmarco-mMiniLMv2-L12-H384-v1`
   (viacjazyčný; číta otázku a chunk spolu), a **prah podobnosti** vyradí chunky so skóre pod −5.
4. **Augmentácia** – prompt obsahuje očíslovaný kontext a pokyn odpovedať len z neho a citovať zdroje.
5. **Generovanie** – lokálny model `gemma3:4b` cez Ollama; bez LLM sa
   zobrazí nájdený kontext. Prompt vyžaduje odpovedať len z kontextu, doslovne uplatniť označenia
   v dokumentoch (napr. *„dishes are vegetarian unless marked vegan“*), citovať zdroje [n]
   a odpovedať v jazyku otázky (jazyk sa deteguje a dá sa modelu ako pokyn v systémovom prompte).

   Porovnanie lokálnych modelov na rovnakých otázkach:

   | Model | Kvalita odpovedí | Slovenčina |
   |---|---|---|
   | `llama3.2` (3B) | pôvodne vypísal všetky vegetariánske cestoviny ako vegánske (opravené promptom) | gramaticky chybná |
   | `qwen2.5:7b` | príliš strohé, neúplné odpovede | – |
   | **`gemma3:4b`** | správne vegánske jedlá, krátke odpovede s citáciami | dobrá → **zvolený** |
6. **Vyhodnotenie** – retrieval (precision@k, hit@k, MRR) a odpovede (LLM-as-judge), kap. 7.

## 3. Ontológia

Ontológia `ontology/restaurant.owl` (WebProtégé, RDF/XML): **41 tried, 10 objektových vlastností,
2 dátové vlastnosti, 1 individuum, 278 trojíc.**

| Oblasť | Triedy |
|---|---|
| Jedlá | Dish → Starter, MainCourse (MeatDish, SeafoodDish, PastaDish), Dessert, Beverage, SignatureDish (CoralPasta, OceanRisotto, TropicalDessert, SeafoodPlatter) |
| Menu | Menu → RegularMenu, KidsMenu, SeasonalMenu, TastingMenu |
| Kuchyne | Cuisine → Italian, European, Asian, Mediterranean |
| Diéty | DietaryOption → Vegetarian, Vegan, GlutenFree, LactoseFree, Halal |
| Podujatia | Event → EntertainmentEvent (LiveMusicEvent, SpecialCelebration), CulinaryEvent (TastingEvent, DinnerEvent) |
| Priestory | DiningPlace → PrivateRoom, PanoramicTable, UnderwaterHall |

Vlastnosti: `servesDish`, `offersMenu`, `hostsEvent`, `hasSignatureDish`, `containsItem`, `includesDish`,
`hasCuisine`, `hasDietaryOption`, `servedAt`, `takesPlaceAt`, `hasPrice` (decimal), `hasDuration` (integer).

## 4. Napojenie RAG na ontológiu

Ontológia sa používa na troch miestach (`rag/ontology.py`):

**a) Verbalizácia ontológie do textu.** Každá trieda, vlastnosť a individuum sa preloží na krátky text
(54 „faktov“), napr.:

> Ontology concept PastaDish (pasta dish) is a class of the restaurant ontology. PastaDish is a kind of MainCourse.
> Indirectly PastaDish is also a Dish, Restaurant. Every PastaDish hasCuisine at least one Cuisine. …

Fakty o konceptoch nájdených v otázke idú do kontextu LLM ako samostatný blok, takže model pozná
hierarchiu a vzťahy domény (napr. že Vegan je DietaryOption a Dish má `hasDietaryOption`).

**b) Rozpoznanie konceptov a rozšírenie dopytu.** V otázke sa hľadajú názvy tried (aj rozdelené CamelCase,
napr. *Kids Menu*) a slovenské/anglické synonymá (`cestovin` → PastaDish, `vegán` → Vegan, `degust` → TastingMenu).
Potom:
- ponechá sa najšpecifickejší koncept (*dish* + *pasta* → PastaDish),
- pridajú sa jeho podtriedy, ak je trieda úzka (≤ 4 podtriedy),
- anglické názvy konceptov sa pridajú k BM25 dopytu a použijú sa ako tretí vektorový dopyt.

Vďaka tomu slovenská otázka *„Koľko stojí degustačné menu?“* hľadá aj anglické termíny *Tasting Menu, Price*
v anglických dokumentoch.

**c) Graf.** Hrany ontológie (subClassOf, reštrikcie, definície, domain→range vlastností, typy individuí)
sa kreslia cez pyvis. V aplikácii sa ku každej odpovedi zobrazí podgraf okolia nájdených konceptov.

**Zistenie pri návrhu:** v ontológii sú všetky triedy podtriedou `Restaurant`. Naivné rozšírenie
o podtriedy by pri slove *restaurant* pridalo celú ontológiu a zhoršilo výsledky. Preto sa podtriedy pridávajú
len pri úzkych triedach a zo všeobecných konceptov sa berie len ich názov (pozri kap. 7).

## 5. Dokumenty (12 PDF)

| Súbor | Obsah | Koncepty ontológie | Zdroj |
|---|---|---|---|
| 01_seafood_restaurant_alc_menu | à la carte menu rybej reštaurácie s cenami | RegularMenu, Starter, SeafoodDish, Dessert, hasPrice | rickstein.com – Seafood Restaurant Sample ALC Menu July 2025 |
| 02_seafood_restaurant_set_menu | menu s pevnou cenou | Menu, SeafoodDish | rickstein.com – Seafood Set Menu 31.03.25 |
| 03_kids_menu | detské menu s označením V / VE | KidsMenu, Vegetarian, Vegan, Dessert | premierinn.com – kids menu |
| 04_tasting_menu_tudor_farmhouse | degustačné menu + párovanie vín | TastingMenu, Beverage, hasPrice | tudorfarmhousehotel.co.uk – Tasting Menu |
| 05_tasting_menu_wine_pairing | degustačné menu s vínami ku chodom | TastingMenu, Beverage | gordonramsayrestaurants.com – GR Las Vegas Tasting Menu |
| 06_italian_vegetarian_menu | talianske vegetariánske menu | Italian, Vegetarian, PastaDish | brioitalian.com – Vegetarian Menu |
| 07_italian_dietary_allergen_guide | diétny a alergénový sprievodca talianskej reštaurácie | Italian, Vegan, LactoseFree, GlutenFree, Halal | vapiano.com – Dietary Allergen Guide |
| 08_vegan_catering_made_easy | vegánske stravovanie v reštauráciách | Vegan | The Vegan Society – Vegan Catering Made Easy |
| 09_halal_caterers_guide | príprava a dohľad nad halal jedlom | Halal | Crescents of Brisbane – The Good Caterer's Guide |
| 10_allergen_technical_guidance | 14 alergénov, povinnosti podnikov | DietaryOption, GlutenFree, LactoseFree | UK FSA – Food allergen labelling technical guidance |
| 11_mediterranean_diet | stredomorská strava | Mediterranean | University of Arkansas Extension – The Mediterranean Diet |
| 12_private_dining_brochure | súkromná sála, oslavy, kapacita, minimum | PrivateRoom, SpecialCelebration, Event | chapteronerestaurant.co.uk – Private Dining Brochure |

Po spracovaní: **233 chunkov z PDF + 54 faktov z ontológie = 287 položiek v indexe.**

## 5a. Webové rozhranie „Ontology Kitchen“

`server.py` (FastAPI) + `web/index.html`. Modely sa načítajú raz pri štarte a zostávajú v pamäti;
nájdené zdroje sa zobrazia hneď (< 1 s) a odpoveď sa vypisuje priebežne (streaming).
Obsahuje: otázku s prepínačmi *Use ontology / Rerank* (porovnanie naživo), odpoveď s klikateľnými
citáciami, zoznam zdrojov so skóre rerankera, nájdené koncepty ontológie, graf ich okolia a celý graf
ontológie, svetlú aj tmavú tému.

## 6. Použité technológie

Python 3.12 · rdflib (OWL) · pypdf · sentence-transformers (embeddingy + cross-encoder reranker) · rank-bm25 ·
NumPy · pyvis / vis-network · FastAPI · Ollama (`gemma3:4b`).

## 7. Vyhodnotenie

`python evaluate.py` – 20 testovacích otázok (14 anglických, 6 slovenských), pri každej je známy PDF
s odpoveďou. Porovnáva sa čistý RAG (len PDF) a RAG s ontológiou, k = 5.

| Nastavenie | hit@5 | precision@5 | MRR |
|---|---|---|---|
| čistý RAG (len PDF) | 0.95 | 0.75 | 0.95 |
| čistý RAG + reranking | 0.90 | 0.76 | 0.88 |
| RAG + ontológia | 0.95 | 0.76 | 0.95 |
| **RAG + ontológia + reranking** | **1.00** | **0.80** | 0.92 |

- **hit@5** – je medzi top-5 aspoň jeden chunk zo správneho dokumentu,
- **precision@5** – podiel top-5 chunkov zo správneho dokumentu,
- **MRR** – priemer 1 / poradie prvého správneho chunku.

**Interpretácia:**
- Najlepšia je kombinácia ontológie a rerankingu: všetkých 20 otázok nájde správny dokument a presnosť
  stúpne na 0.80. Samotný reranking čistému RAG nepomôže (pri slovenských otázkach sa mýli), ale s ontológiou
  dostáva cross-encoder otázku doplnenú o anglické termíny konceptov (napr. *„Aké vegánske cestoviny
  ponúkate? (Pasta Dish, Vegan)“*) – tak sa vyrieši aj jediná dovtedy neúspešná otázka.
- Bez rerankingu ontológia mierne zvyšuje presnosť. Najviac pomáha pri otázkach, kde koncept spája viac slov
  (*„What desserts can children choose?“* → Dessert + KidsMenu: 2 → 4 správne chunky)
  a pri slovenských otázkach (*„Čo patrí do stredomorskej stravy?“*: 4 → 5).
- Navyše LLM dostáva fakty z ontológie (hierarchiu a vzťahy), ktoré v PDF nie sú.
- Prvá verzia bola horšia než čistý RAG (MRR 0.38): fakty ontológie zaberali miesta v top-k a dopyt sa
  rozširoval o **všetky** podtriedy aj nadtriedy – otázka so slovom *restaurant* sa rozšírila na celú ontológiu.
  Pomohlo (1) dávať fakty ontológie do kontextu samostatne, mimo top-k dokumentov, a (2) rozširovať len
  najšpecifickejšie koncepty a len o podtriedy úzkych tried.
- Bez rerankingu zlyhá otázka *„Aké vegánske cestoviny ponúkate?“* – správny dokument (07) je v BM25
  na 1. mieste, ale vektorové vyhľadávanie slovenskej otázky ho radí nízko a v RRF prehrá s detským menu.

### 7.1 Vyhodnotenie odpovedí – LLM-as-judge

`python judge.py --limit 6` – RAG vygeneruje odpoveď (meranie nižšie bolo urobené s `llama3.2`,
pred zmenou promptu a modelu), potom LLM-sudca ohodnotí
odpoveď na škále 1–5:
- **faithfulness** – je všetko v odpovedi podložené kontextom (žiadne halucinácie)?
- **relevance** – odpovedá odpoveď priamo na otázku?

| Nastavenie | faithfulness | relevance |
|---|---|---|
| čistý RAG | 4.83 | 3.83 |
| RAG + ontológia | 4.67 | **4.33** |

(6 otázok, v oboch prípadoch s rerankingom; podrobnosti s odôvodnením sudcu: `output/judge_report.json`.
Sudca je ten istý malý model – výsledky treba brať orientačne.)

- Odpovede sú takmer vždy podložené kontextom (faithfulness ≈ 4.7–4.8) – model necháva „I do not know“,
  keď si nie je istý, namiesto vymýšľania.
- S ontológiou je vyššia relevantnosť. Príklad *„Which seafood starters are on the menu?“*: bez ontológie
  model vypísal aj polievky a šaláty (relevance 2), s ontológiou (koncepty SeafoodDish + Starter)
  vybral morské predjedlá (relevance 5).
- *„How much is the fruits de mer shellfish platter?“* – správny dokument sa nájde, ale malý model
  nevie prečítať tabuľku cien (*„Small 481 kcal 47.50 | Large 797 kcal 85.00“*) a odpovie „I do not know“.
  Ide o obmedzenie 3B modelu, nie vyhľadávania.

## 8. Obmedzenia a možné rozšírenia

- Ontológia má len 1 individuum, preto prináša hlavne **schému** (pojmy a vzťahy); fakty pochádzajú z PDF.
  Riešené rozšírením GraphRAG (kap. 10): individuá sú v samostatnom súbore, ontológia ostala nezmenená.
- Rozpoznávanie konceptov je založené na slovníku synoným; dalo by sa nahradiť embeddingami názvov tried.
- LLM-sudca (kap. 7.1) používa ten istý malý lokálny model, ktorý generuje odpovede – väčší model
  by bol spoľahlivejší sudca.

## 9. Ukážky na obhajobu

```powershell
python ask.py "Which pasta dishes are vegan?"
python ask.py --no-ontology "What desserts can children choose?"     # porovnanie bez ontológie
python ask.py "Koľko stojí degustačné menu?"
python ask.py "How many guests can the private dining room seat?"
python evaluate.py
python judge.py --limit 6
python server.py            # alebo dvojklik na run_app.bat -> http://localhost:8000

# GraphRAG
python build_kg.py          # dishes.json -> restaurant_data.ttl
python evaluate_kg.py       # presnosť SPARQL odpovedí
python compare_graphrag.py  # textový RAG vs. GraphRAG -> output/graphrag_comparison.md
python ask.py "Which vegan dishes cost less than £15?"
python ask.py "Ktoré dezerty sú lacnejšie ako 12 libier?"
```

## 10. Rozšírenie: GraphRAG (znalostný graf + SPARQL)

### 10.1 Prečo
Textový RAG hľadá **úryvky textu**. Na otázky typu *„Ktoré vegánske jedlá stoja menej ako £15?“* nestačí:
ceny sú v jednom PDF, označenia diét v inom a porovnávanie čísel „od oka“ malý model nezvládne.
Riešenie: fakty z menu uložiť ako **znalostný graf** nad našou ontológiou a odpovedať presným dotazom SPARQL.

### 10.2 Dáta
- `data/kg/dishes.json` – **overené (referenčné) dáta** z 8 PDF s menu: text extrahovaný knižnicou pypdf,
  štruktúrované údaje vytvorené s pomocou LLM a ručne overené oproti textu PDF; pri každom jedle je
  zdrojový PDF a strana: **7 reštaurácií, 13 menu, 277 jedál**, ceny (GBP / USD / EUR), kalórie, diéty,
  kódy alergénov, kuchyňa, kategória.
- `extract_kg.py` – **automatická extrakcia** tých istých údajov lokálnym modelom (kap. 10.5) → `dishes_llm.json`.
- `build_kg.py` → `data/kg/restaurant_data.ttl` (RDF/Turtle, 3 354 trojíc). **Ontológia `restaurant.owl` sa
  nemení** – súbor obsahuje len individuá typované triedami ontológie a prepojené jej vlastnosťami
  (`offersMenu`, `containsItem`, `servesDish`, `hasPrice`, `hasCuisine`, `hasDietaryOption`, `hostsEvent`,
  `takesPlaceAt`). Čo ontológia nemá (kalórie, alergény, mena, zdroj, kapacita sály…), je v mennom priestore `kg:`.
- Pravidlá pri stavbe grafu: *vegan ⇒ vegetarian* (podľa legendy „VE = suitable for vegans and vegetarians“);
  pri menu s úplnou legendou alergénov (Chapter One) *bez G ⇒ GlutenFree*, *bez D ⇒ LactoseFree*
  (48 odvodených faktov, označených `kg:dietNote`).

**Zistenia o ontológii pri jej napĺňaní:**
- `SeafoodDish`, `MeatDish`, `PastaDish` sú podtriedy `MainCourse` – morské **predjedlo** sa nedá vyjadriť bez toho,
  aby sa z neho stalo hlavné jedlo. Riešené kategóriou `kg:category "seafood"`.
- `Cuisine` a `DietaryOption` majú podtriedy, ale vlastnosti `hasCuisine` / `hasDietaryOption` potrebujú
  individuá – vytvorené `kg:cuisine_Italian` (typ `Italian`), `kg:diet_Vegan` (typ `Vegan`) atď.

### 10.3 Otázka → SPARQL
Preklad je **deterministický** (bez LLM, `rag/kg.py`), riadený ontológiou:

| V otázke | Rozpoznanie | SPARQL |
|---|---|---|
| *pasta, cestoviny* | trieda `PastaDish` | `?dish a/rdfs:subClassOf* onto:PastaDish` |
| *vegan, vegánske* | trieda `Vegan` | `?dish onto:hasDietaryOption ?d . ?d a onto:Vegan` |
| *Italian, talianske* | trieda `Italian` | `?dish onto:hasCuisine ?c . ?c a onto:Italian` |
| *children, detské* | trieda `KidsMenu` | `?m onto:containsItem ?dish . ?m a onto:KidsMenu` |
| *under £15, lacnejšie ako 12 libier* | regulárny výraz | `FILTER(?price <= 15 && ?currency = "GBP")` |
| *cheapest, najlacnejšie* | regulárny výraz | `ORDER BY ASC(?price) LIMIT n` |
| *without nuts, bez orechov* | slovník alergénov | `FILTER NOT EXISTS { ?dish kg:allergen "nuts" }` |
| *private room* | trieda `PrivateRoom` | dotaz na priestory (kapacita, minimum, záloha) |

Vďaka `rdfs:subClassOf*` sa využíva **hierarchia ontológie**: otázka na `MainCourse` vráti aj individuá
`PastaDish`, `SeafoodDish` a `MeatDish` (odvodenie, ktoré textový RAG nevie).

**Hybrid:** graf sa použije, keď otázka obsahuje jedlo/diétu/menu/priestor/cenu; otázky typu *„ako…“, „prečo…“*
idú len do textového RAG. Výsledok SPARQL ide do kontextu LLM ako samostatný blok *„knowledge graph result“*
(umiestnený tesne pred otázkou – malý model ho tak lepšie dodrží) a v rozhraní sa zobrazí tabuľka,
vygenerovaný SPARQL a nájdené jedlá ako individuá v grafe.

### 10.4 Vyhodnotenie
**Presnosť grafu** (`evaluate_kg.py`): očakávaná odpoveď sa počíta priamo z `dishes.json` obyčajným Python filtrom
(nezávisle od SPARQL) a porovná sa s výsledkom celej cesty otázka → koncepty → SPARQL.

| 14 otázok (EN + SK), 147 očakávaných jedál | precision | recall |
|---|---|---|
| GraphRAG (SPARQL) | **1.00** | **1.00** |

Overuje to správnosť prekladu otázok a dotazov voči dátam – nie správnosť extrakcie z PDF.

**Odpovede LLM** (`compare_graphrag.py`, `gemma3:4b`, 7 otázok so zoznamom jedál, 51 očakávaných jedál;
počíta sa, koľko správnych jedál odpoveď spomenie a koľko nesprávnych):

| | nájdené správne | nesprávne jedlá |
|---|---|---|
| textový RAG | 13 / 51 (25 %) | 18 |
| **GraphRAG** | **37 / 51 (73 %)** | **2** |

Napr. *„Which seafood starters are cheaper than £15?“*: textový RAG 0/16 (vymenoval zeleninové polievky
za £13 – nie sú to morské jedlá), GraphRAG 16/16. Detailné odpovede: `output/graphrag_comparison.md`.

### 10.5 Automatická extrakcia dát lokálnym LLM (`extract_kg.py`)
Aby sa dala naplniť znalostná báza aj bez ručnej práce, `extract_kg.py` posiela text každej strany PDF
(pypdf, rozdelený na úseky ~1 500 znakov) lokálnemu modelu `gemma3:4b` cez Ollama so **štruktúrovaným
výstupom podľa JSON schémy** – trieda jedla môže byť len trieda z ontológie (`Starter`, `PastaDish`, …),
diéty len hodnoty zodpovedajúce triedam `DietaryOption`. Ručne je zadaná len „kostra“ (ktorá strana PDF patrí
ktorému menu / reštaurácii, mena) – 17 riadkov konfigurácie. Výsledok `dishes_llm.json` má rovnaký formát
ako overené dáta, takže z neho `build_kg.py` postaví graf `restaurant_data_llm.ttl`.

Porovnanie s overenými dátami (`output/extraction_report.md`):

| | hodnota |
|---|---|
| nájdené jedlá (recall) | 87 % (240 z 277) |
| presnosť zoznamu (precision) | 61 % (navyše ingrediencie chodov, vína k párovaniu, sýry) |
| správna cena | 87 % |
| správne diéty | 76 % |
| správna trieda ontológie | 77 % |

Ten istý test ako v 10.4, ale nad grafom z automatickej extrakcie (`evaluate_kg.py --ttl …_llm.ttl`):

| graf postavený z | precision | recall |
|---|---|---|
| overených dát | **1.00** | **1.00** |
| automatickej extrakcie `gemma3:4b` | 0.49 | 0.48 |

**Záver:** malý model dobre extrahuje názvy a ceny, ale zlyháva tam, kde diéta nie je pri jedle, ale platí
pre celú sekciu alebo dokument – v menu Brio je „VEGETARIAN MENU“ len raz na konci strany (0/14 správne),
vo Vapiano sú diéty dané zoznamami pod nadpismi (9/38). Preto otázky na vegan / halal / gluten free nad
automatickým grafom zlyhávajú. Pre GraphRAG je kvalita dát rozhodujúca – automatická extrakcia je dobrý
začiatok, ale dáta treba overiť (tak vznikol `dishes.json`). Prvý pokus s úsekmi 2 200 znakov a bez poľa
kuchyne dal podobný výsledok (0.52 / 0.50); posledná strana Vapiano vtedy vrátila neplatný JSON.

**Obmedzenia:** dáta pre graf sa museli extrahovať (8 PDF s menu, nie všetky dokumenty); preklad otázok je
pravidlový – nezvyčajne formulovanú otázku nemusí rozpoznať (potom odpovie textový RAG); malý model pri dlhých
zoznamoch niekedy vynechá položky – úplný zoznam je vždy v tabuľke grafu v rozhraní.
