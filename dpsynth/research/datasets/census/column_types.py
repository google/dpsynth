# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Authoritative column layout and type classification for the 1940 census.

This module is the single source of truth for:

  * ``COLUMN_SPECS`` / ``COLUMN_NAMES`` -- the fixed-width layout of the raw
    IPUMS ``census_full_1940.dat`` extract (147 columns).
  * The type classification of every column into one of NUMERICAL, CATEGORICAL,
    OPEN_SET, or DROP.

The classification drives the derived domain (``build_domain.py``), the proto
schema (``generate_proto.py``) and the marginal discretization
(``precompute_marginals.py``). It intentionally does *not* affect the parquet
representation: ``convert_to_parquet.py`` stores every kept column as its raw
integer IPUMS code, which is lossless and type-agnostic.

Classification philosophy:

  * NUMERICAL -- quantities with a meaningful magnitude / order: ages, years,
    counts, incomes, occupation scores, weeks/hours worked. These are
    discretized into bins downstream (64 bins for the precomputed marginals).
  * OPEN_SET -- codes whose public value set is *not* enumerable from the
    codebook (the IPUMS SAS file ships no value labels for them). Modeled as
    ``OpenSetCategoricalAttribute`` so DP partition selection discovers and
    prunes the tail at synth time.
  * CATEGORICAL -- coded factors with an enumerable codebook value set, from
    low-cardinality (sex, race, marital status) up to high-cardinality but still
    fully-listed columns (CITY, OCC, birthplaces, detailed geography).
    Everything
    kept that is not NUMERICAL or OPEN_SET.
  * DROP -- non-attribute fields excluded from the domain entirely: pure record
    identifiers / bookkeeping (serial numbers, household ids, within-record
    indices) and survey design artifacts (sampling weights, price deflator).

``build_domain.py`` demotes to OPEN_SET only a CATEGORICAL column that has no
codebook labels at all (nothing to enumerate). Size is not a demotion criterion:
if the public codebook lists the values, they are used verbatim regardless of
cardinality.

Separately, ``NON_DOMAIN`` lists columns that are kept in the raw data (parquet)
but excluded from the modeled ``domain.yaml`` -- currently just
the survey ``SAMPLE`` identifier. dpsynth ignores data columns absent from the
domain, so those columns are never measured or synthesized.
"""

import enum

# The raw data is stored in a fixed-width format. ``COLUMN_SPECS`` gives the
# (start, end) byte offsets of each field; ``COLUMN_NAMES`` names them in order.
COLUMN_SPECS = [
    (0, 4),
    (4, 10),
    (10, 18),
    (18, 20),
    (20, 22),
    (22, 32),
    (32, 36),
    (36, 37),
    (37, 39),
    (39, 44),
    (44, 46),
    (46, 48),
    (48, 50),
    (50, 54),
    (54, 55),
    (55, 58),
    (58, 62),
    (62, 66),
    (66, 73),
    (73, 75),
    (75, 76),
    (76, 83),
    (83, 86),
    (86, 89),
    (89, 92),
    (92, 93),
    (93, 94),
    (94, 97),
    (97, 99),
    (99, 100),
    (100, 101),
    (101, 103),
    (103, 107),
    (107, 114),
    (114, 116),
    (116, 117),
    (117, 118),
    (118, 119),
    (119, 120),
    (120, 121),
    (121, 123),
    (123, 132),
    (132, 133),
    (133, 134),
    (134, 142),
    (142, 146),
    (146, 147),
    (147, 151),
    (151, 161),
    (161, 171),
    (171, 172),
    (172, 173),
    (173, 175),
    (175, 177),
    (177, 178),
    (178, 179),
    (179, 180),
    (180, 182),
    (182, 183),
    (183, 184),
    (184, 186),
    (186, 187),
    (187, 188),
    (188, 190),
    (190, 191),
    (191, 192),
    (192, 193),
    (193, 194),
    (194, 196),
    (196, 198),
    (198, 200),
    (200, 204),
    (204, 205),
    (205, 208),
    (208, 210),
    (210, 211),
    (211, 215),
    (215, 216),
    (216, 218),
    (218, 220),
    (220, 221),
    (221, 224),
    (224, 225),
    (225, 228),
    (228, 231),
    (231, 236),
    (236, 239),
    (239, 244),
    (244, 247),
    (247, 252),
    (252, 253),
    (253, 254),
    (254, 256),
    (256, 260),
    (260, 261),
    (261, 262),
    (262, 263),
    (263, 265),
    (265, 268),
    (268, 270),
    (270, 273),
    (273, 274),
    (274, 276),
    (276, 277),
    (277, 278),
    (278, 280),
    (280, 284),
    (284, 287),
    (287, 291),
    (291, 294),
    (294, 296),
    (296, 297),
    (297, 299),
    (299, 300),
    (300, 303),
    (303, 306),
    (306, 309),
    (309, 312),
    (312, 313),
    (313, 319),
    (319, 320),
    (320, 322),
    (322, 324),
    (324, 327),
    (327, 331),
    (331, 335),
    (335, 339),
    (339, 340),
    (340, 342),
    (342, 345),
    (345, 349),
    (349, 353),
    (353, 354),
    (354, 358),
    (358, 361),
    (361, 362),
    (362, 364),
    (364, 365),
    (365, 366),
    (366, 368),
    (368, 369),
    (369, 370),
    (370, 371),
    (371, 372),
    (372, 408),
    (408, 410),
    (410, 411),
]

COLUMN_NAMES = [
    'YEAR',
    'SAMPLE',
    'SERIAL',
    'NUMPREC',
    'SUBSAMP',
    'HHWT',
    'NUMPERHH',
    'HHTYPE',
    'SLPERNUM',
    'CPI99',
    'REGION',
    'STATEICP',
    'STATEFIP',
    'COUNTYICP',
    'METRO',
    'METAREA',
    'METAREAD',
    'CITY',
    'CITYPOP',
    'SIZEPL',
    'URBAN',
    'URBPOP',
    'SEA',
    'WARD',
    'CNTRY',
    'GQ',
    'GQTYPE',
    'GQTYPED',
    'GQFUNDS',
    'FARM',
    'OWNERSHP',
    'OWNERSHPD',
    'RENT',
    'VALUEH',
    'NFAMS',
    'NSUBFAM',
    'NCOUPLES',
    'NMOTHERS',
    'NFATHERS',
    'MULTGEN',
    'MULTGEND',
    'ENUMDIST',
    'RESPOND',
    'SPLIT',
    'SPLITHID',
    'SPLITNUM',
    'EDMISS',
    'PERNUM',
    'PERWT',
    'SLWT',
    'SLREC',
    'RESPONDT',
    'FAMUNIT',
    'FAMSIZE',
    'SUBFAM',
    'SFTYPE',
    'SFRELATE',
    'MOMLOC',
    'STEPMOM',
    'MOMRULE_HIST',
    'POPLOC',
    'STEPPOP',
    'POPRULE_HIST',
    'SPLOC',
    'SPRULE_HIST',
    'NCHILD',
    'NCHLT5',
    'NSIBS',
    'ELDCH',
    'YNGCH',
    'RELATE',
    'RELATED',
    'SEX',
    'AGE',
    'AGEMONTH',
    'MARST',
    'BIRTHYR',
    'MARRNO',
    'AGEMARR',
    'CHBORN',
    'RACE',
    'RACED',
    'HISPAN',
    'HISPAND',
    'BPL',
    'BPLD',
    'MBPL',
    'MBPLD',
    'FBPL',
    'FBPLD',
    'NATIVITY',
    'CITIZEN',
    'MTONGUE',
    'MTONGUED',
    'SPANNAME',
    'HISPRULE',
    'SCHOOL',
    'HIGRADE',
    'HIGRADED',
    'EDUC',
    'EDUCD',
    'EMPSTAT',
    'EMPSTATD',
    'LABFORCE',
    'CLASSWKR',
    'CLASSWKRD',
    'OCC',
    'OCC1950',
    'IND',
    'IND1950',
    'WKSWORK1',
    'WKSWORK2',
    'HRSWORK1',
    'HRSWORK2',
    'DURUNEMP',
    'UOCC',
    'UOCC95',
    'UIND',
    'UCLASSWK',
    'INCWAGE',
    'INCNONWG',
    'OCCSCORE',
    'SEI',
    'PRESGL',
    'ERSCOR50',
    'EDSCOR50',
    'NPBOSS50',
    'MIGRATE5',
    'MIGRATE5D',
    'MIGPLAC5',
    'MIGCOUNTYICP5',
    'MIGMETAREA5',
    'MIGMETRO5',
    'MIGCITY5',
    'MIGSEA5',
    'SAMEPLAC5',
    'VERSIONHIST',
    'SAMESEA5',
    'VETSTAT',
    'VETSTATD',
    'VET1940',
    'VETWWI',
    'VETPER',
    'VETCHILD',
    'HISTID',
    'SURSIM',
    'SSENROLL',
]

assert len(COLUMN_SPECS) == len(
    COLUMN_NAMES
), f'{len(COLUMN_SPECS)} specs vs {len(COLUMN_NAMES)} names'


class ColumnType(enum.Enum):
  """How a column is modeled in the derived domain."""

  NUMERICAL = 'numerical'
  CATEGORICAL = 'categorical'
  OPEN_SET = 'open_set'
  DROP = 'drop'


# Survey weights + price deflator: design artifacts, not person attributes.
# Dropped entirely (folded into DROP below) so they never enter the domain,
# proto, or marginals. precompute_marginals.py also filters them defensively.
WEIGHTS = frozenset({'HHWT', 'PERWT', 'SLWT', 'CPI99'})

# Pure record identifiers / bookkeeping and survey design artifacts. Not
# attributes; excluded from the domain. ``person_id`` (added during conversion)
# is kept in the parquet as a row id but is likewise never an attribute.
DROP = (
    frozenset({
        'SERIAL',  # household serial number
        'SPLITHID',  # split household id
        'SPLITNUM',  # split sequence number
        'SUBSAMP',  # subsample number
        'PERNUM',  # person number within household (index)
        'SLPERNUM',  # sample-line person number (index)
        'HISTID',  # 36-char stable record hash (unique per person)
    })
    | WEIGHTS
)

# Columns retained in the parquet and proto formats (so the already-generated
# data stays valid) but excluded from the *modeled* domain. SAMPLE is the IPUMS
# sample identifier -- a survey-design artifact like the weights, not a person
# attribute -- so it should not be measured or synthesized.
NON_DOMAIN = frozenset({'SAMPLE'})

# Quantities with a meaningful magnitude / order. Discretized into bins
# downstream. All are integer-coded in the raw data (scores carry implied
# decimals but are stored as integers, so ``dtype='int'`` is lossless).
NUMERICAL = frozenset({
    # Household / family counts.
    'NUMPREC',
    'NUMPERHH',
    'NFAMS',
    'NSUBFAM',
    'NCOUPLES',
    'NMOTHERS',
    'NFATHERS',
    'FAMUNIT',
    'FAMSIZE',
    'NCHILD',
    'NCHLT5',
    'NSIBS',
    'CHBORN',
    'MARRNO',
    # Family-pointer person numbers (0 = none).
    'MOMLOC',
    'POPLOC',
    'SPLOC',
    # Ages and years.
    'AGE',
    'AGEMONTH',
    'BIRTHYR',
    'AGEMARR',
    'ELDCH',
    'YNGCH',
    # Geographic magnitudes.
    'CITYPOP',
    'URBPOP',
    'ENUMDIST',
    # Income and housing amounts.
    'INCWAGE',
    'INCNONWG',
    'VALUEH',
    'RENT',
    # Work intensity.
    'WKSWORK1',
    'HRSWORK1',
    'DURUNEMP',
    # Educational attainment (ordinal grade).
    'HIGRADE',
    'HIGRADED',
    # Occupation-based continuous scores.
    'OCCSCORE',
    'SEI',
    'PRESGL',
    'ERSCOR50',
    'EDSCOR50',
    'NPBOSS50',
})

# Codes with no enumerable public value set: the IPUMS SAS codebook ships no
# value labels for these columns, so their possible values cannot be listed
# data-independently. Modeled as open-set so DP partition selection discovers
# and prunes the tail at synth time.
OPEN_SET = frozenset({
    'WARD',  # detailed geography, no codebook
    'UOCC',  # occupation (unrecoded), no codebook
    'UIND',  # industry (unrecoded), no codebook
    'RACED',  # detailed race, no codebook
    'MIGCOUNTYICP5',  # migration county, no codebook
    'MIGSEA5',  # migration SEA, no codebook
})

# Public, data-independent value ranges for the NUMERICAL columns.
NUMERICAL_RANGES: dict[str, tuple[int, int]] = {
    # Household / family counts.
    'NUMPREC': (1, 100),
    'NUMPERHH': (1, 100),
    'NFAMS': (0, 50),
    'NSUBFAM': (0, 50),
    'NCOUPLES': (0, 50),
    'NMOTHERS': (0, 50),
    'NFATHERS': (0, 50),
    'FAMUNIT': (1, 50),
    'FAMSIZE': (1, 60),
    'NCHILD': (0, 20),
    'NCHLT5': (0, 15),
    'NSIBS': (0, 20),
    'CHBORN': (0, 40),
    'MARRNO': (0, 10),
    # Family-pointer person numbers (0 = none / not applicable).
    'MOMLOC': (0, 99),
    'POPLOC': (0, 99),
    'SPLOC': (0, 99),
    # Ages and years.
    'AGE': (0, 120),
    'AGEMONTH': (0, 99),
    'BIRTHYR': (1820, 1940),
    'AGEMARR': (0, 99),
    'ELDCH': (0, 99),
    'YNGCH': (0, 99),
    # Geographic magnitudes (population fields are in thousands).
    'CITYPOP': (0, 10000),
    'URBPOP': (0, 10000),
    'ENUMDIST': (0, 100000),
    # Income and housing amounts (US dollars; 1940 wage topcode $5001+).
    'INCWAGE': (0, 5100),
    'INCNONWG': (0, 9),
    'VALUEH': (0, 100000),
    'RENT': (0, 1000),
    # Work intensity (weeks / hours).
    'WKSWORK1': (0, 52),
    'HRSWORK1': (0, 99),
    'DURUNEMP': (0, 260),
    # Educational attainment (ordinal grade codes).
    'HIGRADE': (0, 30),
    'HIGRADED': (0, 350),
    # Occupation-based scores (the *SCOR50 / NPBOSS50 carry implied decimals).
    'OCCSCORE': (0, 100),
    'SEI': (0, 100),
    'PRESGL': (0, 100),
    'ERSCOR50': (0, 10000),
    'EDSCOR50': (0, 10000),
    'NPBOSS50': (0, 10000),
}

assert set(NUMERICAL_RANGES) == set(NUMERICAL), (
    'NUMERICAL_RANGES must cover exactly the NUMERICAL columns; '
    f'missing={set(NUMERICAL) - set(NUMERICAL_RANGES)}, '
    f'extra={set(NUMERICAL_RANGES) - set(NUMERICAL)}'
)


def column_type(name: str) -> ColumnType:
  """Returns the static (pre-data) classification for ``name``."""
  if name in DROP:
    return ColumnType.DROP
  if name in NUMERICAL:
    return ColumnType.NUMERICAL
  if name in OPEN_SET:
    return ColumnType.OPEN_SET
  return ColumnType.CATEGORICAL


def kept_columns() -> list[str]:
  """Returns the attribute columns (raw order, identifiers excluded)."""
  return [c for c in COLUMN_NAMES if c not in DROP]


def domain_columns() -> list[str]:
  """Returns the columns that enter the modeled domain (kept minus NON_DOMAIN).

  The dataset schemas carry ``kept_columns()``; the domain and the marginals
  are built over this narrower set so survey-design fields like SAMPLE are
  present in the data but never modeled.
  """
  return [c for c in kept_columns() if c not in NON_DOMAIN]
