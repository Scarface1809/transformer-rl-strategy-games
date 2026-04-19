from __future__ import annotations

from typing import Dict

from envs.core.enums import Nation

REWARD_TILES: Dict[Nation, Dict[int, float]] = {
    # GALICIA starts NW (0,1)  →  rewards on E coast (Valencia/Denia belt)
    Nation.GALICIA: {
        39: 1.0,  # Cartagena
        40: 1.0,  # Denia
        41: 1.0,  # Calatrava
        42: 1.0,  # Valencia
        43: 1.0,  # Albarracin
        44: 1.0,  # Castellon
    },
    # CANTABRIA starts N (2,3)  →  rewards in central-south (Toledo basin)
    Nation.CANTABRIA: {
        23: 1.0,  # Toletum
        24: 1.0,  # Estremadura
        31: 1.0,  # Baccula
        32: 1.0,  # Corduba
        33: 1.0,  # Baetica
        37: 1.0,  # Granada
    },
    # BASQUES starts N (4,5)  →  rewards in SW Portugal / Algarve coast
    Nation.BASQUES: {
        27: 1.0,  # Badajoz
        28: 1.0,  # Vetonia
        29: 1.0,  # Algarve
        30: 1.0,  # Onuba
        31: 1.0,  # Baccula
        34: 1.0,  # Sevilla
    },
    # TURDETANS starts SE (37,39)  →  rewards in far NW (Galicia / Asturian coast)
    Nation.TURDETANS: {
        0: 1.0,  # Galicia
        1: 1.0,  # Oporto
        2: 1.0,  # Asturia
        13: 1.0,  # Leon
        14: 1.0,  # Duero
        15: 1.0,  # Termes
    },
    # CARTHAGE starts deep S (35,36)  →  rewards on N Meseta / Ebro headwaters
    Nation.CARTHAGE: {
        9: 1.0,  # Saragossa
        10: 1.0,  # Numantina
        11: 1.0,  # Burgos
        12: 1.0,  # Palencia
        18: 1.0,  # Segovia
        19: 1.0,  # Atienza
    },
    # LUSITANIA starts SW (28,29)  →  rewards on NE coast (Ebro delta / Catalonia)
    Nation.LUSITANIA: {
        7: 1.0,  # Pyrenees Orientalis
        46: 1.0,  # Tarraco
        47: 1.0,  # Illerda
        48: 1.0,  # Barcino
        53: 1.0,  # Septimania
        54: 1.0,  # Aquitania
    },
    # IBERES starts E coast (45,46)  →  rewards in W / central Portugal
    Nation.IBERES: {
        15: 1.0,  # Termes
        17: 1.0,  # Salamanca
        22: 1.0,  # Alcantara
        25: 1.0,  # Lusitania (region)
        26: 1.0,  # Tago
        28: 1.0,  # Vetonia
    },
    # ROME starts NE (47,48)  →  rewards in far S Andalusia (Baetica)
    Nation.ROME: {
        30: 1.0,  # Onuba
        31: 1.0,  # Baccula
        33: 1.0,  # Baetica
        34: 1.0,  # Sevilla
        35: 1.0,  # Gades
        36: 1.0,  # Malaca
    },
}
