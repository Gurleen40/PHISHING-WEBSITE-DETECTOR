"""
URL -> feature-vector extractor for the phishing detection model.

Reproduces (as closely as validated against the training CSV) the 20
features the Random Forest was trained on. Two lookup tables
(tld_legit_prob.json, char_prob.json) and one clipping table
(clip_bounds.json) are built from the original training data (url.csv)
and must ship alongside this file / the model.
"""

import json
import math
import os
import re
from urllib.parse import urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "tld_legit_prob.json")) as f:
    _tld_data = json.load(f)
    TLD_MAP = _tld_data["tld_map"]
    TLD_FALLBACK = _tld_data["fallback"]

with open(os.path.join(BASE_DIR, "char_prob.json")) as f:
    CHAR_PROB = json.load(f)

with open(os.path.join(BASE_DIR, "clip_bounds.json")) as f:
    CLIP_BOUNDS = json.load(f)

FEATURE_ORDER = [
    "URLLength", "DomainLength", "IsDomainIP", "CharContinuationRate",
    "TLDLegitimateProb", "URLCharProb", "TLDLength", "NoOfSubDomain",
    "HasObfuscation", "NoOfObfuscatedChar", "ObfuscationRatio",
    "NoOfLettersInURL", "LetterRatioInURL", "NoOfDegitsInURL", "DegitRatioInURL",
    "NoOfEqualsInURL", "NoOfQMarkInURL", "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL", "SpacialCharRatioInURL",
]

IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
PCT_ENCODED_RE = re.compile(r"%[0-9a-fA-F]{2}")


def _char_type(c):
    if c.isalpha():
        return "L"
    if c.isdigit():
        return "D"
    return "S"


def _longest_run_ratio(s, keyfn):
    """Longest run of same-'type' consecutive chars, divided by length.
    Best validated proxy found for CharContinuationRate (approximate —
    PhiUSIIL's exact internal formula isn't published)."""
    if not s:
        return 0.0
    types = [keyfn(c) for c in s]
    max_run = run = 1
    for i in range(1, len(types)):
        if types[i] == types[i - 1]:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    return max_run / len(s)


def _strip_scheme(url):
    return re.sub(r"^https?://", "", url)


def extract_features(url: str) -> dict:
    """Turn a raw URL string into the 20 raw (pre-scaling) feature values
    the model expects, in FEATURE_ORDER."""
    url = url.strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "http://" + url  # assume http(s) if scheme omitted

    parsed = urlparse(url)
    domain = parsed.netloc.split(":")[0]  # strip port if present
    if domain.startswith("www."):
        domain_for_subdomain_count = domain
    else:
        domain_for_subdomain_count = domain

    is_ip = 1 if IP_RE.match(domain) else 0

    # TLD = last dot-separated label of the domain (unless it's an IP)
    domain_parts = domain.split(".") if domain else []
    tld = "" if is_ip or len(domain_parts) < 2 else domain_parts[-1].lower()
    tld_length = len(tld)

    # NoOfSubDomain: dot-separated labels minus (domain + tld), floored at 0
    # Validated at 100% exact match against the training CSV.
    no_of_subdomain = 0 if is_ip else max(len(domain_parts) - 2, 0)

    letters = sum(c.isalpha() for c in url)
    digits = sum(c.isdigit() for c in url)
    n = len(url)
    eq = url.count("=")
    qm = url.count("?")
    amp = url.count("&")
    special = n - letters - digits
    other_special = special - eq - qm - amp

    # Obfuscation: percent-encoded (%XX) sequences.
    # Validated ~99.5% exact match against the training CSV.
    obf_matches = PCT_ENCODED_RE.findall(url)
    no_of_obf_char = len(obf_matches)
    has_obf = 1 if no_of_obf_char > 0 else 0
    obf_ratio = (no_of_obf_char * 3) / n if n else 0.0

    # CharContinuationRate: approximate (see _longest_run_ratio docstring).
    char_cont_rate = _longest_run_ratio(domain, _char_type)

    # TLDLegitimateProb: exact lookup from training data; unseen TLD ->
    # the dataset-wide mean as a neutral fallback.
    tld_legit_prob = TLD_MAP.get(tld, TLD_FALLBACK)

    # URLCharProb: approximate, built from a character-frequency table
    # over the (scheme-stripped) training corpus. corr ~0.70 with the
    # real column on held-out rows — a proxy, not an exact reproduction.
    stripped = _strip_scheme(url)
    if stripped:
        url_char_prob = sum(CHAR_PROB.get(c, 0.0) for c in stripped) / len(stripped)
    else:
        url_char_prob = 0.0

    raw = {
        "URLLength": n,
        "DomainLength": len(domain),
        "IsDomainIP": is_ip,
        "CharContinuationRate": char_cont_rate,
        "TLDLegitimateProb": tld_legit_prob,
        "URLCharProb": url_char_prob,
        "TLDLength": tld_length,
        "NoOfSubDomain": no_of_subdomain,
        "HasObfuscation": has_obf,
        "NoOfObfuscatedChar": no_of_obf_char,
        "ObfuscationRatio": obf_ratio,
        "NoOfLettersInURL": letters,
        "LetterRatioInURL": letters / n if n else 0.0,
        "NoOfDegitsInURL": digits,
        "DegitRatioInURL": digits / n if n else 0.0,
        "NoOfEqualsInURL": eq,
        "NoOfQMarkInURL": qm,
        "NoOfAmpersandInURL": amp,
        "NoOfOtherSpecialCharsInURL": other_special,
        "SpacialCharRatioInURL": special / n if n else 0.0,
    }

    # Apply the SAME IQR winsorization the notebook applied to the
    # training data before scaling, so live inputs land in the range the
    # model actually learned from. NOTE: for several zero-inflated columns
    # (NoOfDegitsInURL, NoOfEqualsInURL, NoOfQMarkInURL, NoOfAmpersandInURL,
    # NoOfObfuscatedChar, ObfuscationRatio, NoOfSubDomain) the training
    # data's IQR bounds collapsed to a single point (e.g. [0,0] or [1,1]).
    # That means the deployed model was trained with these signals almost
    # entirely flattened out — clipping live URLs the same way keeps this
    # extractor faithful to the trained model, but it also means the model
    # is largely blind to query-string/digit-stuffing/subdomain-count
    # differences. Worth calling out as a known limitation.
    for col, (lower, upper) in CLIP_BOUNDS.items():
        raw[col] = min(max(raw[col], lower), upper)

    return {k: raw[k] for k in FEATURE_ORDER}


if __name__ == "__main__":
    # quick manual check against two rows we know the ground truth for
    for u in ["https://www.southbankmosaics.com", "http://www.teramill.com"]:
        print(u)
        for k, v in extract_features(u).items():
            print(f"  {k}: {v}")
        print()
