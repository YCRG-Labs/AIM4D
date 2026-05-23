"""
Architectural addition: cultural / linguistic similarity edges for Stage 4.

Uses a fixed mapping of countries to broad cultural-linguistic blocs
following Inglehart-Welzel + Huntington civilizational clusters plus
language family. Two countries share an edge if they are in the same bloc.

This is a rough but theoretically-motivated complement to the contiguity,
alliance, and trade edge types already in Stage 4. The motivation is that
authoritarian norm diffusion is well-documented to flow along shared
linguistic / cultural channels in addition to geographic / strategic ones
(Ambrosio 2010, Schmotz-Selvik 2025).

Output: data/cultural_pairs.csv with columns (iso3_a, iso3_b, bloc).
"""

import os
import itertools
import pandas as pd

DATA = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DATA, "cultural_pairs.csv")

BLOCS = {
    "anglo": ["USA", "GBR", "CAN", "AUS", "NZL", "IRL"],
    "lusophone": ["PRT", "BRA", "AGO", "MOZ", "CPV", "GNB", "STP", "TLS"],
    "hispanophone": ["ESP", "MEX", "ARG", "COL", "PER", "CHL", "VEN", "ECU",
                     "GTM", "CUB", "BOL", "DOM", "HND", "PRY", "SLV", "NIC",
                     "CRI", "PAN", "URY"],
    "francophone_eu": ["FRA", "BEL", "LUX", "MCO", "CHE"],
    "francophone_africa": ["SEN", "MLI", "BFA", "NER", "CIV", "GIN", "BEN",
                            "TGO", "GAB", "CAF", "CMR", "COG", "DJI", "MDG",
                            "BDI", "RWA", "TCD", "COD"],
    "arabic_mena": ["DZA", "BHR", "EGY", "IRQ", "JOR", "KWT", "LBN", "LBY",
                    "MAR", "OMN", "QAT", "SAU", "SYR", "TUN", "ARE", "YEM",
                    "PSE", "SDN", "MRT", "SOM"],
    "post_soviet": ["RUS", "BLR", "UKR", "MDA", "ARM", "AZE", "GEO", "KAZ",
                    "KGZ", "TJK", "TKM", "UZB", "EST", "LVA", "LTU"],
    "balkans": ["SRB", "MNE", "HRV", "BIH", "MKD", "ALB", "XKX", "SVN", "BGR", "ROU"],
    "germanic": ["DEU", "AUT", "NLD", "SWE", "NOR", "DNK", "ISL", "FIN"],
    "east_asia": ["CHN", "JPN", "KOR", "PRK", "TWN", "MNG", "HKG"],
    "south_east_asia": ["IDN", "PHL", "VNM", "THA", "MMR", "KHM", "LAO", "MYS", "SGP", "BRN"],
    "south_asia": ["IND", "PAK", "BGD", "LKA", "NPL", "BTN", "MDV", "AFG"],
    "sub_saharan_anglophone": ["ZAF", "KEN", "NGA", "GHA", "UGA", "TZA",
                                "ZMB", "ZWE", "BWA", "MWI", "NAM", "LSO", "SWZ",
                                "ETH", "SLE", "LBR", "GMB", "SSD"],
    "central_europe": ["POL", "CZE", "SVK", "HUN", "SVN"],
    "italian": ["ITA", "SMR", "VAT", "MLT"],
    "scandinavia_alt": ["DNK", "NOR", "SWE", "ISL", "FIN"],
    "turkic": ["TUR", "AZE", "KAZ", "KGZ", "TKM", "UZB"],
    "persian": ["IRN", "AFG", "TJK"],
    "caribbean": ["JAM", "HTI", "DOM", "CUB", "TTO", "GUY", "SUR", "BLZ"],
}


def main():
    pairs = []
    for bloc, members in BLOCS.items():
        for a, b in itertools.combinations(sorted(set(members)), 2):
            pairs.append({"iso3_a": a, "iso3_b": b, "bloc": bloc})
    df = pd.DataFrame(pairs).drop_duplicates(subset=["iso3_a", "iso3_b"])
    df.to_csv(OUT, index=False)
    n_countries = len(set(df["iso3_a"]).union(set(df["iso3_b"])))
    print(f"Wrote {OUT}: {len(df)} cultural pairs across "
          f"{df['bloc'].nunique()} blocs, {n_countries} countries")


if __name__ == "__main__":
    main()
