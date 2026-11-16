# doc-extraction-eval

A small evaluation harness for extracting structured data from documents with LLMs.
It runs a set of synthetic documents (PDFs and scanned images) through two prompt versions
and two Claude models, grades the output field by field, and reports accuracy, token usage,
cost and latency. Every paid Messages API call is checked against a hard spending cap and
a per-run limit and recorded in a local ledger (token counting is free and is neither checked
nor recorded); attempts that may have been billed without a response count at their upper
bound. First results (Claude Haiku 5.5 and Sonnet 5.5, one run) are in `docs/vysledky.md`.
MIT licence.
The rest of this README is in Czech.

## Co měří

Úloha: z dokumentu (smlouva o investici, výpis z účtu, potvrzení pokynu) vytěžit šest polí –
`cislo_smlouvy`, `isin`, `isin_platny`, `castka`, `mena`, `citelne` – jako JSON podle schématu
(`src/llmeval/schema.py`).

Nástroj pustí matici **případy × verze promptu × modely × opakování** (`--reps`) a pro každý
řádek zapíše:

- výsledek po polích a verdikt za případ,
- tokeny z `usage` odpovědi (vstup, čtení cache, výstup, thinking, pokud ho API vrátí),
- cenu volání v USD (Decimal, z ceníku v `data/prices.json`),
- latenci, `stop_reason`, model uvedený v odpovědi, effort, počet pokusů.

Modely: Claude Haiku 5.5 a Claude Sonnet 5.5, oba s `effort: medium`. Volání jdou po jednom,
ne souběžně – kontrola rozpočtu před voláním platí jen tehdy, když žádné jiné neběží.

## Sada dokumentů

20 případů v `data/cases/<id>/` (`document.pdf` + `expected.json`), přehled v
`data/cases/index.csv`. Všechny údaje jsou vymyšlené (generátor se seedem); ISIN mají
skutečnou kontrolní číslici podle ISO 6166, kromě případů, kde je chybná schválně.

| případy | záludnost |
|---|---|
| c01–c05 | bez záludnosti |
| c06 | ISIN vytištěný po skupinách s mezerami |
| c07, c08 | ISIN s chybnou kontrolní číslicí (`isin_platny=false`) |
| c09, c10 | částka s mezerami mezi tisíci a „Kč“ |
| c11–c13 | víc částek v dokumentu, platí jedna (opravená / konečný zůstatek / objem bez poplatku) |
| c14, c15 | chybí číslo smlouvy (`null`), u c15 je jen číslo pokynu |
| c16 | peněžní účet bez cenného papíru (`isin` null) |
| c17–c19 | obrázkový sken bez textové vrstvy: mírně rozmazaný; natočený; natočený a zašuměný |
| c20 | sken zničený rozostřením a šumem: `citelne=false`, ostatní pole null |

c17 je poškozený jen slabě. Je to test čtení skenu bez textové vrstvy, ne test odolnosti vůči
rozmazání.

Sadu lze znovu vygenerovat (viz Instalace). Textový obsah PDF a `expected.json` jsou
deterministické; pixely skenů se mohou lišit podle verze `pdftoppm`/Pillow.

## Prompty

Oba prompty dostávají stejné schéma. Jeho popisy polí určují **formát** hodnot (kanonická
částka `"1234.50"`, ISIN 12 znaků velkými písmeny bez mezer, kód měny ISO 4217) a u
`isin_platny` se ptají, zda je kontrolní číslice ISIN platná.

`prompts/v1.md` je jen zadání úlohy: vyjmenuje pole (včetně platnosti kontrolní číslice ISIN)
a odkáže na schéma. `prompts/v2.md` má totéž zadání a k němu pravidla po polích:

- znovu, textem, požadavky na formát ze schématu (kanonická částka, ISIN bez mezer, kód ISO 4217),
- pravidla, která ve schématu nejsou: číslo smlouvy ne číslo pokynu ani účtu; ISIN uvést i
  s chybnou kontrolní číslicí; platnost podle algoritmu ISO 6166; která částka je hlavní podle
  typu dokumentu (u opravené platí opravená); „Kč“ = CZK; nečitelný dokument ⇒ všechna ostatní
  pole null.

Rozdíl v1/v2 tedy měří hlavně pravidla mimo schéma, ale může v něm být i vliv toho, že v2
formát zopakuje v textu.

## Hodnocení

Grader (`src/llmeval/grade.py`) výstup modelu **nenormalizuje**: hodnota ve tvaru, který schéma
nežádá, je chyba i tehdy, když znamená správnou věc. Stav pole:

| stav | význam |
|---|---|
| `ok` | hodnota sedí |
| `spatne` | jiná hodnota, nebo `null` tam, kde hodnota je |
| `format` | špatný typ nebo tvar (`"1.234,50"`, ISIN s mezerami, `""` místo `null`) |
| `chybi_klic` | klíč ve výstupu chybí |
| `navic_hodnota` | hodnota tam, kde má být `null` (vymyšlený údaj) |

`""`, `null` a chybějící klíč jsou tři různé výsledky. Případ projde, když všechna pole jsou
`ok`, výstup nemá klíče navíc a volání skončilo normálně (`end_turn`); odpověď useknutá na
`max_tokens` nebo odmítnutá (`refusal`) neprojde, ani když se její text náhodou dá ohodnotit.
Neplatný JSON = neúspěšný případ s důvodem. Za neplatný se počítá i duplicitní klíč (běžný
parser by potichu vzal poslední hodnotu) a `NaN`/`Infinity`, které JSON nezná.

Úspěšnost se měří při `max_tokens` 4096 **včetně přemýšlení**; tento limit je pevná podmínka
srovnání, ne ladicí parametr. Řádek v sekci „Stopped at max_tokens“ znamená, že model vyčerpal
výstupní rozpočet (typicky přemýšlením) dřív, než odpověděl – ne že dokument špatně přečetl.

Referenční body ve stejném běhu (jen v souhrnu, bez volání API):

- **oracle** vrací `expected.json` – strop, musí dát 100 %,
- **all-null** tvrdí „čitelné, nic nenalezeno“ – dolní mez; na této sadě 0/20, protože každý
  případ má aspoň jedno pole, které all-null netrefí.

## Cena a rozpočet

**Ceník** je v `data/prices.json` s odkazem na zdroj (stránka Pricing v dokumentaci Claude) a
datem ověření 2026-11-16. Ceny se mění; čísla z běhu platí k tomuto datu ceníku.

- Cena se počítá v Decimal z `usage`. Neznámý model je chyba, nikdy cena 0. Nulová cena
  vstupu nebo výstupu v ceníku je chyba ceníku.
- Haiku 5.5 má nad 100 000 tokenů promptu dražší sazbu; nástroj takový prompt odmítne už před
  voláním (podle `count_tokens`) a po volání ho hlásí jako chybu.
- Prompt caching se nepoužívá; nenulový zápis do cache v `usage` je chyba běhu. Chybou je i
  příplatek, který ceník nepopisuje (`inference_geo` jiné než global, jiná rychlost nebo
  service tier, serverové nástroje).
- Thinking tokeny jsou (podle dokumentace) součástí `output_tokens`; samostatně se jen vypisují.

**Strop útraty je 5 USD celkem** – počítá se z celého ledgeru, ne z jednoho běhu. Do stropu
se nepočítá jen součet `cost_usd`: řádek s `"billed": "unknown"` (pokus bez odpovědi, viz níž)
se započítá svou horní mezí `bound_usd`.

1. **Odhad před během:** pro každý případ × prompt × model `count_tokens` se stejnými parametry
   jako ostré volání, k tomu rezerva 10 % + 100 tokenů a výstup v plné výši `max_tokens`
   (výchozí 4096), vše × `--reps`. Rezerva je odhad; ve 83 voláních naměřených 16. 11. 2026
   se fakturovaný vstup shodoval s `count_tokens` přesně (spočteno z `docs/results/`), jinde
   ověřená není. Opakované pokusy v odhadu nejsou.
2. Když ledger (jak se počítá do stropu) + odhad přesáhne 5 USD, běh se vůbec nespustí.
3. Odhad nad 1 USD vyžaduje `--potvrzuji <částka>` s částkou aspoň ve výši odhadu (odhad se
   vypisuje zaokrouhlený nahoru, vypsanou částku tedy stačí opsat).
4. **Před každým voláním** (i před každým opakováním) se ledger znovu přečte a běh skončí
   s částečnými výsledky, když by horní mez dalšího volání překročila
   - strop: ledger (jak se počítá do stropu) + horní mez > 5 USD, nebo
   - **limit běhu** max(1 USD, `--potvrzuji`): co tento běh zapsal (`cost_usd`) + horní meze
     jeho pokusů bez odpovědi + horní mez dalšího volání.
5. **Po každém pokusu** se do ledgeru zapíše řádek se skutečnou cenou z `usage`.

Opakování (nejvýš 3 pokusy, při chybě spojení, timeoutu, 408, 409, 429 a 5xx) dělá nástroj
sám, ne SDK (`max_retries=0`), aby každé prošlo kontrolou rozpočtu. Pokus bez odpovědi se
zapíše s `cost_usd` 0, `"billed": "unknown"` a horní mezí `bound_usd` – zda se takový pokus
účtuje, není ověřeno. Do stropu i do limitu běhu se proto počítá horní mezí; souhrn ji uvádí
zvlášť jako „possible unrecorded spend“. Opakované timeouty tak zastaví běh nejpozději na
limitu běhu a do stropu se počítají i v dalších bězích, dokud řádky někdo nevyřeší.

Když fakturovaný prompt přesáhne `count_tokens` i s rezervou (horní mez, proti které se
kontrolovalo, pak byla nízká), běh se nezastaví, ale řádek dostane varování
`prompt_over_estimate` (sloupec `warnings`, sekce Warnings v souhrnu, výpis na stderr).

### Ledger útraty

Výchozí cesta `~/.local/share/llmeval/spend.jsonl` (jinak `--ledger <cesta>`), mimo repozitář.
Append-only JSONL, řádek na pokus API (čas, běh, model, výsledek, `cost_usd`, tokeny,
`request_id`, …). Pravidla:

- Chybějící, nečitelný nebo poškozený ledger reálný běh zastaví. Nový prázdný ledger vytvoří
  jen `python -m llmeval run --init-ledger`, a to **odmítne, pokud soubor existuje**. Jiný
  přepínač běhu než `--ledger` odmítne kódem 2, pokud nezůstal na výchozí hodnotě.
- **Začít od nuly = ručně smazat soubor** a pak `--init-ledger`. Je to vědomý krok: zmizí tím
  záznam o dosavadní útratě a strop 5 USD se počítá znovu od nuly.
- Běh drží na ledgeru zámek (`flock`); druhý běh nad stejným ledgerem se odmítne.
- Ledger během běhu neupravujte. Běh se zastaví, když součet, který se počítá do stropu, klesne
  pod to, co v ledgeru bylo na začátku plus co běh zapsal, nebo když je soubor nahrazen jiným
  (přejmenováním, jak to dělá `sed -i` a některé editory). Úprava na místě, která součet
  nesníží, zachycena není; chování konkrétních editorů ověřené není.
- Placené volání, jehož cenu nejde určit, se zapíše horní mezí s `"cost_known": false` a další
  volání v témže běhu se už nepustí.
- **Řádky `"billed": "unknown"` vyřešit ručně mezi běhy.** Dokud zůstanou, snižují zbývající
  rozpočet o svou horní mez. Po kontrole skutečné spotřeby (např. v Anthropic Console) řádek
  upravte: `"billed": "no"` (a `cost_usd` `"0"`), pokud pokus účtovaný
  nebyl, nebo `"billed": "yes"` se skutečnou cenou v `cost_usd`. Ledger tehdy nesmí držet žádný
  běh (úprava za běhu, která sníží součet, běh zastaví). Souhrn uvádí, kolik takových řádků
  v ledgeru zbývá.
- Řádek, u kterého nejde určit, kolik se počítá do stropu – `billed` jiné než `yes`/`no`/
  `unknown`, `unknown` bez platného nenulového `bound_usd` (desetinné číslo jako řetězec),
  `no` s nenulovou cenou, `yes` s cenou 0 u pokusu bez odpovědi (`"outcome": "no_response"`),
  klíč uvedený v řádku dvakrát – je chyba ledgeru: zastaví běh se zprávou s číslem řádku.
  Řádek bez klíče `billed` platí jako `yes`: počítá se svým `cost_usd` a u pokusu bez
  odpovědi s cenou 0 je chybou stejně jako `yes`.
- Když se ledger změní kolem zápisu pokusu nebo zápis selže, běh skončí a souhrn řekne jedno
  ze tří:
  - `attempt written to the ledger … do not add it by hand` – pokus v ledgeru je, nic nedoplňovat;
  - `add this line to … by hand: …` nebo `unrecorded attempt: …` – pokus v ledgeru chybí,
    doplnit uvedený řádek (u `unrecorded attempt` je `attempt_id` uvedené, pokud už vzniklo);
  - `unknown whether the attempt is in the ledger …` nebo `… look for attempt_id … before adding
    it by hand` (např. přerušení během samotného zápisu) – nejdřív v ledgeru vyhledat
    `attempt_id`, teprve pak případně doplnit.

  Řiďte se zprávou v souhrnu a před ručním doplněním vždy zkontrolujte `attempt_id`.

`--fake` ledger nečte, nezamyká ani do něj nezapisuje; přepínače reálného běhu (`--models`,
`--max-tokens`, `--potvrzuji`, `--ledger`, `--init-ledger`) s ním skončí kódem 2.

## Instalace

Požadavky: Python 3.12+ a mimo pip:

- `poppler-utils` – `pdftoppm` (skeny), `pdftotext` a `pdfinfo` (testy),
- `img2pdf` (skeny),
- písmo DejaVu Sans (`fonts-dejavu-core`), hledá se v `/usr/share/fonts/truetype/dejavu`,
  `/usr/share/fonts/TTF` a `/usr/share/fonts/dejavu`; jinde nastavte `LLMEVAL_FONT_DIR` na
  adresář s `DejaVuSans.ttf` a `DejaVuSans-Bold.ttf`.

Na Debianu/Ubuntu: `sudo apt install poppler-utils img2pdf fonts-dejavu-core`.

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

Nástroj čte `data/` a `prompts/` relativně ke zdrojovým souborům, funguje proto jen z klonu
repozitáře s editovatelnou instalací (`-e`), ne jako běžně nainstalovaný balíček.

Znovu vygenerovat sadu dokumentů do `data/cases/`:

```bash
.venv/bin/python -m llmeval.dataset.build
```

## Spuštění

Kontrola kódu a testy (ruff a pytest; testy nesmějí na síť, pytest běží s `--disable-socket`):

```bash
./qa.sh
```

Běh s falešným modelem – bez API, bez ledgeru, zdarma. Vrací připravené výstupy
z `data/fake/outputs.json` (část schválně chybných), které nezávisí na promptu; slouží
k ověření celé cesty hodnocení a reportu:

```bash
.venv/bin/python -m llmeval run --fake
```

Reálný běh (stojí kredit; potřebuje přihlášení k API, např. `ANTHROPIC_API_KEY`):

```bash
.venv/bin/python -m llmeval run --init-ledger            # jen poprvé
.venv/bin/python -m llmeval run --models haiku --cases c01 c17
.venv/bin/python -m llmeval run --potvrzuji <odhad>      # celá matice, 1 opakování
```

Odhad je horní mez, ne předpověď útraty. U této sady v něm převažuje výstup v plné výši
`max_tokens` (4096) × cena výstupu: v běhu 16. 11. 2026 tvořil 85,6 % odhadu (spočteno, viz
níž). Spočteno z horní meze s `count_tokens`
nahrazeným nulou (bez API): celá matice s 1 opakováním vychází nejméně na 1,7288 USD (vyžaduje
`--potvrzuji`), se `--reps 3` nejméně na 5,1862 USD, takže ji strop odmítne ještě před prvním
voláním; samotný Haiku s 1 opakováním nejméně 0,0824 USD. Skutečné tokeny promptu odhad dál
zvyšují: v běhu 16. 11. 2026 byl odhad celé matice s 1 opakováním 2,0108 USD, z toho výstup
v plné výši `max_tokens` 1,7203 USD (spočteno: 40 × 4096 tokenů za cenu výstupu každého
modelu), a běh stál 0,46892 USD (`docs/vysledky.md`). Rozsah se zužuje přes `--models`,
`--cases` a `--max-tokens`.

Další přepínače: `--models haiku sonnet` (nebo ID modelu z `data/prices.json`; jiné ID skončí
kódem 2), `--prompts v1 v2`, `--cases <id…>`, `--reps N`, `--max-tokens N`,
`--results-dir <adresář>` (výchozí `results/`), `--ledger <cesta>`.
Úplný výpis: `python -m llmeval run --help`. Příkazy pouštějte z kořene repozitáře: výchozí
`results/` i výstup generátoru `data/cases` jsou relativní k aktuálnímu adresáři.

Kód ukončení:

| kód | význam |
|---|---|
| 0 | běh doběhl (i když některé případy neprošly) |
| 1 | běh se zastavil uprostřed (strop, chyba API, která se neopakuje, chyba ceny nebo ledgeru); výsledky jsou částečné |
| 2 | odmítnuto před prvním voláním: chybný argument (i přepínač reálného běhu u `--fake`), chybějící/zamčený/poškozený ledger, `--init-ledger` nad existujícím souborem nebo s dalším přepínačem běhu, selhaný odhad (i neznámé ID modelu), strop, chybějící nebo nízké `--potvrzuji` |
| 130 | přerušeno Ctrl+C; částečné výsledky se zapíší. Padne-li přerušení do zápisu pokusu do ledgeru, zůstane 130 a souhrn má poznámku, co s pokusem udělat (doplnit, nebo nejdřív hledat `attempt_id`) |
| 1 (po Ctrl+C) | přerušení padlo do volání API a zápis pokusu do ledgeru pak selhal nebo se ledger mezitím změnil tak, že řádek chybí nebo je nejistý – souhrn říká, co doplnit |

## Výstupy

Každý běh vytvoří `results/<run id>/`. Run id je čas spuštění (`2026-11-16T12-00-00`), při
obsazeném názvu s příponou (`-2`, `-3`, …); stejné id je v hlavičce `summary.md` i v poli
`run_id` řádků ledgeru, které běh zapsal.

- `runs.jsonl` – řádek po řádku průběžně, přežije přerušení,
- `runs.csv` – totéž jako tabulka,
- `summary.md` – souhrn (vypíše se i do terminálu).

Řádek = případ × prompt × model × opakování. Vybrané sloupce `runs.csv`:

- `status` – jak volání skončilo (`ok`, `max_tokens`, `refusal`, `unexpected_stop`,
  `no_response` = žádný ze tří pokusů nedostal odpověď); `passed` – verdikt řádku,
- `raw` – surový text odpovědi modelu,
- `<pole>_expected`, `<pole>_actual`, `<pole>_status` – hodnoty jako JSON (`null`, `""` a text
  jdou rozlišit); prázdná buňka v `_actual` znamená chybějící klíč, nebo – při prázdném
  `_status` a `fields_counted=false` – že výstup vůbec není JSON objekt,
- `fields_counted` – `false`, když volání neskončilo normálně nebo výstup není JSON objekt;
  souhrn pak takový řádek počítá jako chybu ve všech polích, ať `*_status` říká cokoli
  (to je verdikt graderu nad samotným textem),
- `input_tokens`, `cache_read_input_tokens`, `output_tokens`, `thinking_tokens`, `cost_usd`,
  `latency_s`, `stop_reason`, `response_model`, `effort`, `attempts`,
  `prompt_tokens_estimate` (z `count_tokens`), `request_id`,
- `warnings` – varování k ruční kontrole, verdikt nemění: `response_model_differs` (odpověď
  uvádí jiný model, než se žádal; cena se počítá podle žádaného), `prompt_over_estimate`
  (fakturovaný prompt nad `count_tokens` s rezervou).

`runs.jsonl` má tytéž sloupce s typy JSON: `passed`, `fields_counted` jako boolean, tokeny a
latence jako čísla, `cost_usd` jako desetinný řetězec, chybějící hodnota jako `null`,
`warnings` jako seznam. Sloupce `<pole>_*` zůstávají jako v CSV.

`summary.md` obsahuje v hlavičce run id, commit nástroje (`git rev-parse HEAD` v klonu, ze
kterého běží, s poznámkou při necommitnutých změnách sledovaných souborů; `unknown` bez gitu
nebo když kód neběží z kořene klonu, který jeho soubory sleduje – např. instalace mimo klon,
kopie uvnitř jiného repozitáře nebo soubory rozbalené do kořene jiného repozitáře), režim,
prompty, případy (při `--cases` jejich seznam), opakování a datum ověření ceníku; u reálného
běhu i modely, effort, `max_tokens`, odhad, limit běhu a cestu k ledgeru (domovský adresář jako
`~`, stejně jako v hlášení o zastavení běhu). Dál: úspěšnost po model × prompt (s cenou celkem,
na úspěšný řádek, na řádek a mediánem latence), průměrné vstupní a výstupní tokeny na řádek po
model × prompt, úspěšnost po polích, tabulku po případech, referenční body oracle/all-null,
párové srovnání v2 proti v1 po případech (zlepšeno / zhoršeno / beze změny), seznam useknutých
na `max_tokens`, varování, seznam selhání s důvodem a útratu (zapsáno do ledgeru tímto během,
možná nezapsaná, ledger: kolik se počítá do stropu, součet `cost_usd` a počet nevyřešených řádků
`"billed": "unknown"`).

**Řádky vs. případy:** úspěšnost a úspěšnost po polích počítají řádky (případ × opakování),
ne případy. Při `--reps 3` je na 20 případů 60 řádků. Tabulka po případech ukazuje počet
úspěšných opakování (např. `2/3`) a párové srovnání porovnává podíl úspěšných opakování
každého případu.

Ukázka z běhu **s falešným modelem** (`--fake`; čísla popisují připravené fixtury, ne model):

```
| model | prompt | passed rows | cost USD | cost per passed row | mean cost per row | median latency s |
|---|---|---|---|---|---|---|
| fake | v1 | 12/20 (60%) | - | - | - | 0.00 |
| fake | v2 | 12/20 (60%) | - | - | - | 0.00 |

| baseline | passed |
|---|---|
| oracle | 20/20 (100%) |
| all-null | 0/20 (0%) |
```

## Omezení

- **Malá sada, šum.** 20 případů, 1–3 na každou záludnost. Rozdíl jednoho dvou případů mezi
  prompty nebo modely může být náhoda; souhrn to u párového srovnání připomíná.
- **Syntetická data.** Dokumenty jsou generované a stejně formátované; o reálných skenech a
  rozmanitých šablonách výsledky mnoho neříkají. Skeny jsou čtyři a c17 je poškozený jen slabě.
- **Nedeterministický výstup.** Sampling parametry (`temperature` apod.) se neposílají, u
  modelů 5.5 je API podle dokumentace odmítá; proto `--reps`.
- **Effort.** Oba modely běží s `effort: medium`. Je to stejná nominální úroveň, ne stejné
  množství přemýšlení – úrovně nejsou mezi modely sjednocené (Sonnet 5.5 je má podle
  dokumentace proti předchozí verzi překalibrované). `max_tokens` 4096 včetně přemýšlení je
  pevná podmínka srovnání; Sonnet na c17 jednou vyčerpal celý limit přemýšlením
  (`docs/vysledky.md`). Useknuté odpovědi souhrn vypisuje zvlášť.
- **Ceny k datu.** Ceník je ručně opsaný ke dni v `data/prices.json`.
- **Změřeno 16. 11. 2026 (jeden plný běh a tři kalibrační, reps 1; 83 volání, data v
  `docs/results/`; počty spočtené napříč jejich `runs.jsonl`, platí jen pro tato volání):**
  - API vrátilo `thinking_tokens` u obou modelů (vyplněné ve všech 83 voláních);
  - effort medium přemýšlel i bez parametru `thinking`, u Haiku i Sonnetu (nenulové thinking
    v 78 z 83 volání);
  - fakturovaný vstup se rovnal `count_tokens` ve všech 83 voláních;
  - `model` v odpovědi se rovnal žádanému aliasu ve všech 83 voláních;
  - API přijalo schéma výstupu i `effort: medium` u obou modelů.
- **Neověřené předpoklady o účtování:** rezerva odhadu 10 % + 100 tokenů proti fakturovaným
  tokenům u jiných dokumentů a promptů než změřených (překročení se jen hlásí varováním); zda
  se účtují pokusy bez odpovědi (timeout, přerušení) – v měřených bězích žádný takový pokus
  nenastal; cena je spočtená z `usage` a ceníku, s vyúčtováním v Anthropic Console porovnaná
  není.
- **Baseline all-null** je hrubá dolní mez (0/20), ne „hloupý model“; slouží jen k ověření, že
  skóre nevzniká ze samých `null`.
- Falešný model nezávisí na promptu, takže párové srovnání v `--fake` jen prověří report.

## Výsledky

První měření (16. 11. 2026, Haiku 5.5 a Sonnet 5.5, effort medium, jeden běh s 1 opakováním)
je v `docs/vysledky.md`, surová data všech běhů v `docs/results/` (s vyprázdněným
`request_id`). Stručně: 79 z 80 řádků prošlo, takže sada je u stropu a modely ani prompty
nerozliší; celý běh stál 0,46892 USD.

## Licence

MIT, viz `LICENSE`.
