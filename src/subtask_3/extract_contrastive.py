# =========================
# COLAB / PATH SETUP
# =========================

import sys
import os

PROJECT_ROOT = "/content/HCC_dimABSA"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# =========================
# STANDARD IMPORTS
# =========================

import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

from src.subtask_1.train_subtask1 import main as train_reg
from src.shared import config


# =========================
# DOMAIN-SPECIFIC CATEGORY MAP
# =========================

ENTITY_ATTRIBUTE_MAP = {
    "restaurant": {
        "ENTITY": [
            "RESTAURANT", "FOOD", "DRINKS", "AMBIENCE", "SERVICE", "LOCATION"
        ],
        "ATTRIBUTE": [
            "GENERAL", "PRICES", "QUALITY", "STYLE_OPTIONS", "MISCELLANEOUS"
        ]
    },
    "laptop": {
        "ENTITY": [
            "LAPTOP", "DISPLAY", "KEYBOARD", "MOUSE", "MOTHERBOARD", "CPU",
            "FANS_COOLING", "PORTS", "MEMORY", "POWER_SUPPLY", "OPTICAL_DRIVES",
            "BATTERY", "GRAPHICS", "HARD_DISK", "MULTIMEDIA_DEVICES",
            "HARDWARE", "SOFTWARE", "OS", "WARRANTY", "SHIPPING",
            "SUPPORT", "COMPANY"
        ],
        "ATTRIBUTE": [
            "GENERAL", "PRICE", "QUALITY", "DESIGN_FEATURES",
            "OPERATION_PERFORMANCE", "USABILITY",
            "PORTABILITY", "CONNECTIVITY", "MISCELLANEOUS"
        ]
    },
    "hotel": {
        "ENTITY": [
            "HOTEL", "ROOMS", "FACILITIES", "ROOM_AMENITIES",
            "SERVICE", "LOCATION", "FOOD_DRINKS"
        ],
        "ATTRIBUTE": [
            "GENERAL", "PRICE", "COMFORT", "CLEANLINESS",
            "QUALITY", "DESIGN_FEATURES",
            "STYLE_OPTIONS", "MISCELLANEOUS"
        ]
    },
    "finance": {
        "ENTITY": [
            "MARKET", "COMPANY", "BUSINESS", "PRODUCT"
        ],
        "ATTRIBUTE": [
            "GENERAL", "SALES", "PROFIT",
            "AMOUNT", "PRICE", "COST"
        ]
    }
}

DOMAIN = config.DOMAIN.lower()
ENTITY_SET = ENTITY_ATTRIBUTE_MAP[DOMAIN]["ENTITY"]
ATTRIBUTE_SET = ENTITY_ATTRIBUTE_MAP[DOMAIN]["ATTRIBUTE"]

VALID_CATEGORIES_
