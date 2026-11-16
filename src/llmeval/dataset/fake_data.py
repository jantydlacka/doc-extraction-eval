"""Fictional test cases: document content plus the values a careful reader extracts from it.

All names, contract numbers, account numbers and ISINs are generated; the company and funds are
invented. Each case has its own ``random.Random`` seeded by a string, so changing one case does
not shift the others.
"""

import random
from dataclasses import dataclass
from decimal import Decimal

from llmeval.dataset.isin import is_valid_isin, make_isin
from llmeval.schema import canonical_amount

SEED = "doc-extraction-eval/v1"

COMPANY = "Vzorová investiční společnost, a.s."
FUNDS = (
    "Vzorový dluhopisový fond",
    "Vzorový akciový fond Globál",
    "Vzorový smíšený fond Rovnováha",
    "Vzorový fond peněžního trhu",
)
MALE = (("Petr", "Jan", "Tomáš", "Martin", "Lukáš"), ("Doležal", "Kratochvíl", "Holub", "Šimek"))
FEMALE = (("Eva", "Jana", "Lucie", "Veronika", "Marie"), ("Doležalová", "Holubová", "Šimková"))
FOOTER = "Fiktivní dokument vytvořený pro testování; neobsahuje skutečné údaje."

SMLOUVA = "smlouva_o_investici"
VYPIS = "vypis_z_uctu"
POKYN = "potvrzeni_pokynu"


@dataclass(frozen=True)
class Block:
    kind: str  # title, subtitle, heading, field, para, note
    text: str
    value: str = ""

    def line(self) -> str:
        return f"{self.text}: {self.value}" if self.kind == "field" else self.text


@dataclass(frozen=True)
class Document:
    blocks: tuple[Block, ...]

    def lines(self) -> list[str]:
        return [b.line() for b in self.blocks]


@dataclass(frozen=True)
class ScanParams:
    dpi: int = 100
    blur: float = 0.0
    angle: float = 0.0
    noise: int = 0  # peak amplitude of uniform noise, in grey levels
    contrast: float = 1.0
    downscale: int = 1  # shrink and enlarge again by this factor to destroy detail
    jpeg_quality: int = 70


@dataclass(frozen=True)
class Case:
    id: str
    typ: str
    forma: str  # "text" or "sken"
    stitky: tuple[str, ...]
    zaludnost: str
    document: Document
    expected: dict
    castka_text: str | None  # the main amount as printed, None when unreadable
    scan: ScanParams | None = None


@dataclass(frozen=True)
class _Opts:
    mena: str = "CZK"
    castka_styl: str = (
        "kod"  # kod: "48 500,00 CZK"; kc: "1 250 000,50 Kč"; kc_pomlcka: "35 000,- Kč"
    )
    isin: str = "ok"  # ok, spatna_cislice, s_mezerami, zadny
    cislo_smlouvy: bool = True
    dve_castky: bool = False
    velka_castka: bool = False


def _thousands(value: Decimal) -> str:
    whole, _, cents = f"{value:.2f}".partition(".")
    groups = []
    while whole:
        groups.insert(0, whole[-3:])
        whole = whole[:-3]
    return " ".join(groups) + "," + cents


def format_amount(value: Decimal, mena: str, styl: str) -> str:
    if styl == "kod":
        return f"{_thousands(value)} {mena}"
    if styl == "kc":
        return f"{_thousands(value)} Kč"
    if styl == "kc_pomlcka":
        if value != value.to_integral_value():
            raise ValueError("',-' only fits whole amounts")
        return f"{_thousands(value)[:-3]},- Kč"
    raise ValueError(styl)


def _date(rng: random.Random) -> str:
    return f"{rng.randint(1, 28)}. {rng.randint(1, 10)}. 2026"


def _person(rng: random.Random) -> str:
    first, last = rng.choice((MALE, FEMALE))
    return f"{rng.choice(first)} {rng.choice(last)}"


def _contract_number(rng: random.Random) -> str:
    return f"SM-2026-{rng.randrange(10**6):06d}"


def _amount(rng: random.Random, opts: _Opts, low: int, high: int, cents: bool) -> Decimal:
    if opts.velka_castka:
        low, high = 1_000_000, 3_000_000
    if opts.mena == "EUR":
        low, high = low // 25, high // 25
    value = Decimal(rng.randrange(low // 100, high // 100) * 100)
    if cents:
        value += Decimal(rng.randrange(1, 100)) / 100
    return value


def _isin(rng: random.Random, opts: _Opts) -> tuple[str | None, str | None]:
    """Return (value for expected, text as printed)."""
    if opts.isin == "zadny":
        return None, None
    # Country CZ with a "99" prefix keeps the code in an invented-looking range.
    isin = make_isin("CZ", "99" + f"{rng.randrange(10**7):07d}")
    if opts.isin == "spatna_cislice":
        isin = isin[:11] + str((int(isin[11]) + rng.randint(1, 9)) % 10)
    printed = isin
    if opts.isin == "s_mezerami":
        printed = " ".join(isin[i : i + 4] for i in range(0, 12, 4))
    return isin, printed


def _header(title: str) -> list[Block]:
    return [Block("title", title), Block("subtitle", COMPANY)]


def _smlouva(rng: random.Random, opts: _Opts):
    number = _contract_number(rng)
    isin, isin_text = _isin(rng, opts)
    amount = _amount(rng, opts, 10_000, 500_000, cents=False)
    fund = rng.choice(FUNDS)
    printed = format_amount(amount, opts.mena, opts.castka_styl)
    blocks = _header("Smlouva o investici") + [
        Block("heading", "Smluvní strany"),
        Block("field", "Investiční společnost", COMPANY),
        Block("field", "Investor", _person(rng)),
        Block("heading", "Údaje o investici"),
    ]
    if opts.cislo_smlouvy:
        blocks.append(Block("field", "Číslo smlouvy", number))
    else:
        number = None
        blocks.append(
            Block("para", "Návrh smlouvy. Číslo smlouvy bude přiděleno po podpisu oběma stranami.")
        )
    blocks += [
        Block("field", "Fond", fund),
        Block("field", "ISIN", isin_text),
        Block("field", "Výše jednorázové investice", printed),
        Block("field", "Vstupní poplatek", "1,5 % z investované částky"),
    ]
    if opts.dve_castky:
        corrected = amount + Decimal(rng.randrange(1, 50) * 1000)
        printed = format_amount(corrected, opts.mena, "kod")
        blocks.append(
            Block(
                "para",
                f"Oprava: výše jednorázové investice se mění na {printed}. "
                "Částka uvedená výše neplatí.",
            )
        )
        amount = corrected
    if number is not None:
        blocks.append(Block("field", "Datum uzavření", _date(rng)))
    blocks += [
        Block(
            "para",
            "Investor prohlašuje, že se seznámil se statutem fondu a se sdělením klíčových "
            "informací a že investiční částku uhradí do 10 dnů od uzavření smlouvy.",
        ),
    ]
    return blocks, number, amount, printed, isin


def _vypis(rng: random.Random, opts: _Opts):
    number = _contract_number(rng)
    isin, isin_text = _isin(rng, opts)
    amount = _amount(rng, opts, 20_000, 800_000, cents=True)
    title = "Výpis z peněžního účtu" if isin is None else "Výpis z investičního účtu"
    blocks = _header(title) + [
        Block("field", "Období", "1. 7. 2026 – 30. 9. 2026"),
        Block("field", "Majitel účtu", _person(rng)),
        Block("field", "Číslo účtu", f"IU-{rng.randrange(10**8):08d}"),
        Block("field", "Smlouva č.", number),
        Block("heading", "Stav účtu"),
    ]
    if isin is not None:
        units = Decimal(rng.randrange(10**6, 10**8)) / 10**4
        blocks += [
            Block("field", "Fond", rng.choice(FUNDS)),
            Block("field", "ISIN", isin_text),
            Block("field", "Počet podílových listů", f"{units:.4f}".replace(".", ",") + " ks"),
        ]
    printed = format_amount(amount, opts.mena, opts.castka_styl)
    if opts.dve_castky:
        start = amount - Decimal(rng.randrange(1000, 9000)) - Decimal(rng.randrange(1, 100)) / 100
        blocks += [
            Block(
                "field",
                "Počáteční zůstatek k 1. 7. 2026",
                format_amount(start, opts.mena, opts.castka_styl),
            ),
            Block("field", "Konečný zůstatek k 30. 9. 2026", printed),
        ]
    else:
        label = "Konečný zůstatek" if isin is None else "Hodnota investice"
        blocks.append(Block("field", f"{label} k 30. 9. 2026", printed))
    blocks.append(Block("para", "Výpis slouží pouze pro informaci majitele účtu."))
    return blocks, number, amount, printed, isin


def _pokyn(rng: random.Random, opts: _Opts):
    number = _contract_number(rng)
    isin, isin_text = _isin(rng, opts)
    amount = _amount(rng, opts, 5_000, 200_000, cents=False)
    blocks = _header("Potvrzení pokynu k nákupu") + [
        Block("field", "Číslo pokynu", f"PK-{rng.randrange(10**8):08d}"),
    ]
    if opts.cislo_smlouvy:
        blocks.append(Block("field", "Číslo smlouvy", number))
    else:
        number = None
    printed = format_amount(amount, opts.mena, opts.castka_styl)
    blocks += [
        Block("field", "Investor", _person(rng)),
        Block("field", "Datum přijetí pokynu", _date(rng)),
        Block("field", "Fond", rng.choice(FUNDS)),
        Block("field", "ISIN", isin_text),
    ]
    if opts.dve_castky:
        fee = (amount * Decimal("0.015")).quantize(Decimal("0.01"))
        blocks += [
            Block("field", "Objem pokynu (bez poplatku)", printed),
            Block("field", "Vstupní poplatek 1,5 %", format_amount(fee, opts.mena, "kod")),
            Block("field", "Celkem k úhradě", format_amount(amount + fee, opts.mena, "kod")),
        ]
    else:
        blocks.append(Block("field", "Objem pokynu", printed))
    blocks.append(
        Block(
            "para",
            "Pokyn bude vypořádán za cenu podílového listu platnou ke dni přijetí peněžních "
            "prostředků na účet fondu.",
        )
    )
    return blocks, number, amount, printed, isin


_BUILDERS = {SMLOUVA: _smlouva, VYPIS: _vypis, POKYN: _pokyn}

_TIP_ISIN = "ISIN má chybnou kontrolní číslici, isin_platny=false."
_TIP_KC = "Částka s mezerami mezi tisíci a se značkou Kč místo kódu měny."
_TIP_SCAN = "Obrázkový sken bez textové vrstvy"

# id, typ, forma, stitky, zaludnost, options, scan parameters
_CASES = [
    ("c01", SMLOUVA, (), "", _Opts(), None),
    ("c02", SMLOUVA, (), "", _Opts(mena="EUR"), None),
    ("c03", VYPIS, (), "", _Opts(), None),
    ("c04", VYPIS, (), "", _Opts(mena="EUR"), None),
    ("c05", POKYN, (), "", _Opts(), None),
    (
        "c06",
        POKYN,
        ("isin_s_mezerami",),
        "ISIN je vytištěný po skupinách s mezerami.",
        _Opts(isin="s_mezerami"),
        None,
    ),
    (
        "c07",
        SMLOUVA,
        ("neplatna_kontrolni_cislice",),
        _TIP_ISIN,
        _Opts(isin="spatna_cislice"),
        None,
    ),
    ("c08", POKYN, ("neplatna_kontrolni_cislice",), _TIP_ISIN, _Opts(isin="spatna_cislice"), None),
    (
        "c09",
        VYPIS,
        ("castka_mezery_kc",),
        _TIP_KC + " Částka přes milion s haléři.",
        _Opts(castka_styl="kc", velka_castka=True),
        None,
    ),
    (
        "c10",
        SMLOUVA,
        ("castka_mezery_kc",),
        _TIP_KC + " Zápis „,-“ místo haléřů.",
        _Opts(castka_styl="kc_pomlcka"),
        None,
    ),
    (
        "c11",
        SMLOUVA,
        ("dve_castky",),
        "Dvě částky: dokument původní výši investice opravuje, platí opravená.",
        _Opts(dve_castky=True),
        None,
    ),
    (
        "c12",
        VYPIS,
        ("dve_castky",),
        "Dvě částky: počáteční a konečný zůstatek, platí konečný.",
        _Opts(dve_castky=True),
        None,
    ),
    (
        "c13",
        POKYN,
        ("dve_castky",),
        "Tři částky: objem pokynu, poplatek a celkem k úhradě; platí objem bez poplatku.",
        _Opts(dve_castky=True),
        None,
    ),
    (
        "c14",
        SMLOUVA,
        ("chybi_cislo_smlouvy",),
        "Návrh smlouvy, číslo ještě nebylo přiděleno: cislo_smlouvy=null.",
        _Opts(cislo_smlouvy=False),
        None,
    ),
    (
        "c15",
        POKYN,
        ("chybi_cislo_smlouvy",),
        "Číslo smlouvy chybí, je tam jen číslo pokynu: cislo_smlouvy=null.",
        _Opts(cislo_smlouvy=False),
        None,
    ),
    (
        "c16",
        VYPIS,
        ("bez_isin",),
        "Peněžní účet bez cenného papíru: isin a isin_platny null.",
        _Opts(isin="zadny"),
        None,
    ),
    (
        "c17",
        SMLOUVA,
        ("sken_rozmazany",),
        _TIP_SCAN + ", rozmazaný, ale čitelný.",
        _Opts(),
        ScanParams(blur=1.0, noise=12, contrast=0.85),
    ),
    (
        "c18",
        POKYN,
        ("sken_natoceny",),
        _TIP_SCAN + ", natočený o několik stupňů.",
        _Opts(),
        ScanParams(angle=3.0, noise=10, blur=0.4),
    ),
    (
        "c19",
        VYPIS,
        ("sken_natoceny", "sken_sum"),
        _TIP_SCAN + ", mírně natočený a zašuměný.",
        _Opts(),
        ScanParams(angle=-1.5, noise=45, blur=0.5, contrast=0.8),
    ),
    (
        "c20",
        SMLOUVA,
        ("sken_necitelny",),
        _TIP_SCAN + ", zničený rozostřením a šumem: citelne=false, ostatní pole null.",
        _Opts(),
        ScanParams(downscale=8, blur=4.0, noise=60, contrast=0.4, jpeg_quality=40),
    ),
]


def _expected(number, isin, amount, mena) -> dict:
    return {
        "cislo_smlouvy": number,
        "isin": isin,
        "isin_platny": None if isin is None else is_valid_isin(isin),
        "castka": canonical_amount(amount),
        "mena": mena,
        "citelne": True,
    }


def build_cases(seed: str = SEED) -> list[Case]:
    cases = []
    for case_id, typ, stitky, zaludnost, opts, scan in _CASES:
        rng = random.Random(f"{seed}:{case_id}")
        blocks, number, amount, printed, isin = _BUILDERS[typ](rng, opts)
        blocks.append(Block("note", FOOTER))
        expected = _expected(number, isin, amount, opts.mena)
        if "sken_necitelny" in stitky:
            expected = {**dict.fromkeys(expected), "citelne": False}
            printed = None
        cases.append(
            Case(
                id=case_id,
                typ=typ,
                forma="text" if scan is None else "sken",
                stitky=stitky,
                zaludnost=zaludnost,
                document=Document(tuple(blocks)),
                expected=expected,
                castka_text=printed,
                scan=scan,
            )
        )
    return cases
