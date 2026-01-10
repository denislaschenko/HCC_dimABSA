
import json
import argparse
import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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
