from __future__ import annotations

from typing import Dict

from envs.core.enums import Nation

REWARD_TILES: Dict[Nation, Dict[int, float]] = {
    # GALICIA starts NW (0,1)  →  rewards on E coast (Valencia/Denia belt)
    Nation.GALICIA: {
        39: 5.0,  # Cartagena
        40: 5.0,  # Denia
        41: 5.0,  # Calatrava
        42: 5.0,  # Valencia
        43: 5.0,  # Albarracin
        44: 5.0,  # Castellon
    },
    # CANTABRIA starts N (2,3)  →  rewards in central-south (Toledo basin)
    Nation.CANTABRIA: {
        23: 5.0,  # Toletum
        24: 5.0,  # Estremadura
        31: 5.0,  # Baccula
        32: 5.0,  # Corduba
        33: 5.0,  # Baetica
        37: 5.0,  # Granada
    },
    # BASQUES starts N (4,5)  →  rewards in SW Portugal / Algarve coast
    Nation.BASQUES: {
        27: 5.0,  # Badajoz
        28: 5.0,  # Vetonia
        29: 5.0,  # Algarve
        30: 5.0,  # Onuba
        31: 5.0,  # Baccula
        34: 5.0,  # Sevilla
    },
    # TURDETANS starts SE (37,39)  →  rewards in far NW (Galicia / Asturian coast)
    Nation.TURDETANS: {
        0: 5.0,  # Galicia
        1: 5.0,  # Oporto
        2: 5.0,  # Asturia
        13: 5.0,  # Leon
        14: 5.0,  # Duero
        15: 5.0,  # Termes
    },
    # CARTHAGE starts deep S (35,36)  →  rewards on N Meseta / Ebro headwaters
    Nation.CARTHAGE: {
        9: 5.0,  # Saragossa
        10: 5.0,  # Numantina
        11: 5.0,  # Burgos
        12: 5.0,  # Palencia
        18: 5.0,  # Segovia
        19: 5.0,  # Atienza
    },
    # LUSITANIA starts SW (28,29)  →  rewards on NE coast (Ebro delta / Catalonia)
    Nation.LUSITANIA: {
        7: 5.0,  # Pyrenees Orientalis
        46: 5.0,  # Tarraco
        47: 5.0,  # Illerda
        48: 5.0,  # Barcino
        53: 5.0,  # Septimania
        54: 5.0,  # Aquitania
    },
    # IBERES starts E coast (45,46)  →  rewards in W / central Portugal
    Nation.IBERES: {
        15: 5.0,  # Termes
        17: 5.0,  # Salamanca
        22: 5.0,  # Alcantara
        25: 5.0,  # Lusitania (region)
        26: 5.0,  # Tago
        28: 5.0,  # Vetonia
    },
    # ROME starts NE (47,48)  →  rewards in far S Andalusia (Baetica)
    Nation.ROME: {
        30: 5.0,  # Onuba
        31: 5.0,  # Baccula
        33: 5.0,  # Baetica
        34: 5.0,  # Sevilla
        35: 5.0,  # Gades
        36: 5.0,  # Malaca
    },
}
