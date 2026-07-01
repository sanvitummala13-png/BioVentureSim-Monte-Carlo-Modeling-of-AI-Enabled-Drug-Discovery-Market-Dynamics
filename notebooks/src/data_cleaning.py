"""
data_cleaning.py
================
Flatten the nested ClinicalTrials.gov v2 JSON into a tidy, analysis-ready
DataFrame and persist it to data/processed/clinicaltrials_ai_trials.csv.

Output columns:
    nct_id, brief_title, phase, status, start_date, completion_date,
    conditions, interventions, sponsor, enrollment, study_type,
    query_term, year
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

OUTPUT_COLUMNS = [
    "nct_id", "brief_title", "phase", "status", "start_date",
    "completion_date", "conditions", "interventions", "sponsor",
    "enrollment", "study_type", "query_term", "year",
]


def _safe_get(d: dict, *keys, default=None):
    """Walk a nested dict via successive keys, returning default if missing."""
    cur: Any = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def _parse_year(date_str: str | None) -> int | None:
    """Extract a 4-digit year from a 'YYYY-MM' / 'YYYY' style string."""
    if not date_str:
        return None
    try:
        return int(str(date_str)[:4])
    except (ValueError, TypeError):
        return None


def _flatten_study(study: dict[str, Any]) -> dict[str, Any]:
    """Convert one nested study record into a flat row dict."""
    ps = study.get("protocolSection", {})

    nct_id = _safe_get(ps, "identificationModule", "nctId")
    brief_title = _safe_get(ps, "identificationModule", "briefTitle")
    phases = _safe_get(ps, "designModule", "phases", default=[]) or []
    phase = ", ".join(phases) if phases else "NA"
    status = _safe_get(ps, "statusModule", "overallStatus")
    start_date = _safe_get(ps, "statusModule", "startDateStruct", "date")
    completion_date = _safe_get(
        ps, "statusModule", "completionDateStruct", "date"
    )

    conditions_list = _safe_get(ps, "conditionsModule", "conditions", default=[]) or []
    conditions = "; ".join(conditions_list)

    interventions_list = _safe_get(
        ps, "armsInterventionsModule", "interventions", default=[]
    ) or []
    interventions = "; ".join(
        i.get("name", "") for i in interventions_list if isinstance(i, dict)
    )

    sponsor = _safe_get(ps, "sponsorCollaboratorsModule", "leadSponsor", "name")
    enrollment = _safe_get(ps, "designModule", "enrollmentInfo", "count")
    study_type = _safe_get(ps, "designModule", "studyType")
    query_term = study.get("_query_term")
    year = _parse_year(start_date)

    return {
        "nct_id": nct_id,
        "brief_title": brief_title,
        "phase": phase,
        "status": status,
        "start_date": start_date,
        "completion_date": completion_date,
        "conditions": conditions,
        "interventions": interventions,
        "sponsor": sponsor,
        "enrollment": enrollment,
        "study_type": study_type,
        "query_term": query_term,
        "year": year,
    }


def clean_clinical_trials(raw: dict[str, Any]) -> pd.DataFrame:
    """
    Turn the raw API payload into a clean DataFrame, drop duplicate trials
    (the same NCT id can match several query terms), and save to processed/.
    """
    studies = raw.get("studies", [])
    rows = [_flatten_study(s) for s in studies]
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # Type coercion
    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # Keep the first appearance of each trial, but remember it can map to
    # multiple query terms; we preserve the term that first surfaced it.
    df = df.drop_duplicates(subset="nct_id").reset_index(drop=True)

    config.ensure_dirs()
    df.to_csv(config.PROCESSED_TRIALS_CSV, index=False)
    logger.info(
        "Cleaned %d unique trials -> %s", len(df), config.PROCESSED_TRIALS_CSV
    )
    return df


if __name__ == "__main__":
    from src.data_fetch import fetch_clinical_trials

    clean_clinical_trials(fetch_clinical_trials())

