# Výsledky

First measured results: Claude Haiku 5.5 and Claude Sonnet 5.5 (effort medium, `max_tokens` 4096
including thinking) on 20 synthetic documents with two prompt versions, one run, one repetition.
79 of 80 rows passed, so this set is at its ceiling and does not separate the models or the
prompts; total cost 0.46892 USD. The rest of this page is in Czech.

Všechna čísla níže pocházejí ze souborů v `docs/results/<run id>/` (`summary.md`, `runs.csv`,
`runs.jsonl`); u každé tabulky je uvedeno, ze kterého. Hodnoty označené **spočteno** v žádném
souboru nejsou – odvodil jsem je z uvedených čísel a postup je napsaný u nich. Surová data jsou
v repozitáři celá, jen s vyprázdněným `request_id`.

## Podmínky

Zdroj: hlavička `docs/results/2026-11-16T22-14-40/summary.md`.

| | |
|---|---|
| datum běhu | 16. 11. 2026 |
| commit nástroje | `da2ee3954ec0aace762f0d544cca670c7a4fca55` – commit z vývoje před vydáním, ve veřejné historii není; kód nástroje (`src/`, `data/`, `prompts/`, `pyproject.toml`, `qa.sh`) je s vydaným commitem shodný (ověření: `git rev-parse HEAD:src` = `59a00aa567860e70b16770d674ebf46e9111af6a`) |
| modely | `claude-haiku-5-5`, `claude-sonnet-5-5` |
| effort | `medium` (oba modely) |
| `max_tokens` | 4096, **včetně přemýšlení**; stanoveno před celým během jako pevná podmínka srovnání; stejné ve všech čtyřech bězích včetně kalibrace |
| prompty | v1, v2 |
| případy | 20 (všechny) |
| opakování | 1 (`--reps 1`) |
| ceník ověřen | 2026-11-16 (`data/prices.json`) |

## Celý běh (2026-11-16T22-14-40)

Zdroj: `docs/results/2026-11-16T22-14-40/summary.md`, tabulky *Pass rate* a *Tokens per row*.

| model | prompt | prošlo | cena USD | cena na úspěšný řádek | průměr vstupních tokenů | průměr výstupních tokenů | medián latence s |
|---|---|---|---|---|---|---|---|
| Haiku 5.5 | v1 | 20/20 | 0,013061 | 0,000653 | 2806,0 | 744,9 | 4,33 |
| Haiku 5.5 | v2 | 19/20 | 0,013920 | 0,000733 | 3298,0 | 732,4 | 4,51 |
| Sonnet 5.5 | v1 | 20/20 | 0,211020 | 0,010551 | 2806,0 | 493,9 | 4,66 |
| Sonnet 5.5 | v2 | 20/20 | 0,230920 | 0,011546 | 3298,0 | 495,0 | 4,53 |

Výstupní tokeny zahrnují thinking (poznámka u tabulky *Tokens per row*). Běh měl 80 řádků
(20 případů × 2 prompty × 2 modely × 1 opakování), trval 384,8 s (wall time) a stál celkem
**0,46892 USD** (*Spend*: „cost of the graded calls: 0.468920 USD“). Odhad nástroje před
během byl **2,0108 USD** (hlavička: „estimate: up to 2.0108 USD“), tedy zhruba 4,3× víc, než
běh stál (**spočteno**: 2,0108 / 0,46892). Odhad je schválně horní mez: výstup každého volání
počítá v plné výši `max_tokens` (4096), skutečný průměr byl 494–745 tokenů. Na `max_tokens` se
v celém běhu nezastavilo žádné volání (*Stopped at max_tokens*: none), varování žádná
(*Warnings*: none).

Jediné selhání (*Failures*): c03, prompt v2, Haiku – `isin_platny: spatne (False, expected
True)`. Model u platného ISIN odpověděl, že kontrolní číslice platná není.

## Co z toho plyne a co ne

**Sada je u stropu.** 79 z 80 řádků prošlo (**spočteno**: součet sloupce *passed rows*);
oracle dává 20/20, all-null 0/20 (*Baselines*).
Takový výsledek modely ani prompty nerozliší. Je to zjištění o téhle sadě dokumentů – v tomto
běhu byla pro oba modely příliš snadná –, ne o tom, že jsou modely stejně dobré obecně.

**Sonnet proti Haiku:** na této sadě Sonnet nepřinesl měřitelný rozdíl v úspěšnosti (v1 20/20
oba; v2 20/20 proti 19/20).

**v2 proti v1:** v2 nepřinesla měřitelný rozdíl. Párové srovnání po případech (*Prompts against
v1, paired per case*): Haiku 0 zlepšeno, 1 zhoršeno (c03), 19 beze změny; Sonnet 0 / 0 / 20.
Souhrn sám připomíná, že rozdíl jednoho dvou případů při 1 opakování a nedeterministickém
výstupu může být šum. Jeden zhoršený případ proto neznamená, že v2 je horší.

**Cena.** Sazby Sonnetu jsou proti Haiku 20× vyšší na vstupu i na výstupu (`data/prices.json`:
vstup 2 vs. 0,10, výstup 10 vs. 0,50 USD za milion tokenů). Změřená cena je ale jen asi 16×
vyšší (**spočteno**: (0,211020 + 0,230920) / (0,013061 + 0,013920) ≈ 16,4; po promptech 16,2
a 16,6). Důvod je v tokenech: v tomto běhu měl Haiku víc výstupních tokenů – průměrně 744,9 a
732,4 na řádek proti 493,9 a 495,0 u Sonnetu (*Tokens per row*), u v1 zhruba 1,5×
(**spočteno**). Většinu výstupu tvořil thinking: z `runs.jsonl` celého běhu (součet
`thinking_tokens` / součet `output_tokens`) to bylo u Haiku asi 91 %, u Sonnetu asi 87 %
(**spočteno**).

**Cena v2.** v2 přidává 492 vstupních tokenů na volání (**spočteno**: 3298,0 − 2806,0, u obou
modelů stejně). U Haiku byla v2 v tomto běhu celkem o 6,6 % dražší (**spočteno**:
0,013920 / 0,013061 − 1), u Sonnetu o 9,4 % (**spočteno**: 0,230920 / 0,211020 − 1). Rozdíl
v2 − v1 jde prakticky celý za delším promptem (**spočteno** ze součtů `input_tokens` a
`output_tokens` v `runs.jsonl` celého běhu a sazeb z `data/prices.json`):

| model | rozdíl ceny v2 − v1 USD | z toho vstup | z toho výstup |
|---|---|---|---|
| Haiku 5.5 | +0,0008585 | +0,000984 (+9840 tokenů) | −0,0001255 (−251 tokenů) |
| Sonnet 5.5 | +0,019900 | +0,019680 (+9840 tokenů) | +0,000220 (+22 tokenů) |

U Sonnetu je to asi 99 % rozdílu (**spočteno**). U Haiku měla v2 o něco méně výstupu, což
rozdíl zmenšilo; delší prompt sám by cenu zvýšil o 7,5 % (**spočteno**: 0,000984 / 0,013061).
Kolik z rozdílu ve výstupu je náhoda, při jednom opakování nejde říct.

## Kalibrace

Před celým během proběhla tři samostatná volání, každé jako vlastní běh. Zdroj: `runs.jsonl` a
`summary.md` v `docs/results/<run id>/`. „Vstup“ je fakturovaný `input_tokens`, „odhad vstupu“
je `prompt_tokens_estimate` z `count_tokens`; „horní mez“ je řádek „estimate: up to“ v hlavičce
`summary.md`.

| run id | model | případ | prompt | vstup / odhad vstupu | výstup / z toho thinking | výsledek | cena USD | horní mez USD |
|---|---|---|---|---|---|---|---|---|
| 2026-11-16T21-51-45 | Haiku 5.5 | c01 | v1 | 2885 / 2885 | 702 / 634 | prošlo (`end_turn`) | 0,000640 | 0,0024 |
| 2026-11-16T21-56-53 | Sonnet 5.5 | c17 | v1 | 2595 / 2595 | 4096 / 4095 | neprošlo: `max_tokens`, prázdná odpověď | 0,046150 | 0,0469 |
| 2026-11-16T22-01-58 | Sonnet 5.5 | c01 | v1 | 2885 / 2885 | 470 / 402 | prošlo (`end_turn`) | 0,010470 | 0,0476 |

Ve všech třech voláních se fakturovaný vstup přesně shodoval s `count_tokens`.

**Sonnet na c17 vyčerpal limit.** Ve volání 2026-11-16T21-56-53 spotřeboval celý `max_tokens`
(4096 výstupních tokenů, z toho 4095 thinking) a odpověď zůstala prázdná (`raw` je prázdný
řetězec). V celém běhu stejný případ se stejným promptem prošel s 520 výstupními tokeny (452
thinking), s v2 prošel s 534 (466 thinking) (`runs.jsonl` celého běhu, řádky c17 Sonnet). Za
všechny čtyři běhy tedy Sonnet na c17 s v1 narazil na limit v 1 ze 2 volání, s v2 v 0 z 1
(**spočteno**: počet řádků c17 Sonnet ve všech čtyřech `runs.jsonl`).
Jak často se to stává, z toho určit nejde – jen to, že se to stát může. Pevný `max_tokens`
4096 je proto součástí podmínek srovnání: úspěšnost platí pro tento limit.

**Další pozorování (všechna volání všech čtyř běhů, 83 řádků v `runs.jsonl`).** Všechny počty
v tomto seznamu jsou **spočtené** napříč čtyřmi soubory `runs.jsonl` (1 + 1 + 1 + 80 řádků) a
platí jen pro tato volání:

- Effort medium v těchto voláních přemýšlel i bez parametru `thinking` (nástroj ho neposílá):
  nenulové `thinking_tokens` mělo 78 z 83 volání, u obou modelů. Pět volání s nulovým thinking
  jsou případy c16 (bez ISIN) a c20 (nečitelný sken) – Sonnet c16 v1 a v2, c20 v1 a v2; Haiku
  c20 v2 –, všechna prošla.
- API vrátilo `thinking_tokens` ve všech 83 voláních (hodnota vyplněná v každém řádku).
- Model uvedený v odpovědi (`response_model`) se shodoval se žádaným aliasem ve všech 83
  voláních; varování `response_model_differs` nepadlo.
- Všech 83 volání proběhlo na první pokus (`attempts` 1), bez čtení z cache.

## Omezení

- **Jeden běh, 1 opakování.** Sampling parametry se neposílají a výstup není deterministický;
  stejné volání může příště dopadnout jinak (c17 se Sonnetem a v1 jednou neprošlo a jednou
  prošlo, viz výše). Rozdíl jednoho případu proto nelze brát jako efekt.
- **20 případů, syntetická vymyšlená data.** Dokumenty vytvořil jeden generátor
  (`src/llmeval/dataset/`): tři typy (smlouva o investici, výpis z účtu, potvrzení pokynu),
  každý z jedné šablony s obměnami pro záludné případy, vše stejným písmem; 16 textových PDF
  a 4 obrázkové skeny (`data/cases/index.csv`). O reálných dokumentech a skenech z rozmanitých
  šablon výsledky mnoho neříkají.
- **Sada je u stropu**, takže neměří rozdíl mezi modely ani prompty. Na srovnání by byla potřeba
  těžší sada.
- **Ceny k 2026-11-16.** Cena je počítaná z `usage` a ceníku v `data/prices.json`, ne z
  vyúčtování v Anthropic Console.
- **Effort medium** je stejná nominální úroveň, ne stejné množství přemýšlení: v tomto běhu měl
  Haiku víc thinking tokenů než Sonnet (viz Cena výše).
