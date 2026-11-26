"""Academic productivity tools for student planning and schedule calculation."""

from __future__ import annotations

import datetime
import re
from typing import Any
from langchain_core.tools import tool


def _parse_flexible_date(date_str: str) -> datetime.date | None:
    """Attempt parsing a date string using common academic date formats."""
    cleaned = date_str.strip()
    formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %d %Y",
        "%b %d %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%Y/%m/%d",
    ]
    for fmt in formats:
        try:
            return datetime.datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


@tool
def days_until_deadline(deadline: str) -> str:
    """Calculate the number of days remaining until an academic deadline."""
    if not isinstance(deadline, str) or not deadline.strip():
        return "Error: Deadline must be a non-empty date string (e.g., 'YYYY-MM-DD' or 'Month DD, YYYY')."

    target_date = _parse_flexible_date(deadline)
    if target_date is None:
        return f"Error: Unable to parse deadline '{deadline.strip()}'. Please use a format like 'YYYY-MM-DD' or 'Month DD, YYYY'."

    today = datetime.date.today()
    delta_days = (target_date - today).days

    if delta_days > 1:
        return f"Deadline: {target_date.strftime('%B %d, %Y')} ({delta_days} days remaining)."
    elif delta_days == 1:
        return f"Deadline: {target_date.strftime('%B %d, %Y')} (Tomorrow, 1 day remaining)."
    elif delta_days == 0:
        return f"Deadline: {target_date.strftime('%B %d, %Y')} is TODAY! All work should be submitted before midnight."
    else:
        passed_days = abs(delta_days)
        return f"Deadline: {target_date.strftime('%B %d, %Y')} passed {passed_days} day{'s' if passed_days != 1 else ''} ago."


@tool
def calculate_study_schedule(topics: str, available_hours: float) -> str:
    """Distribute available study hours evenly across requested academic topics."""
    if not isinstance(topics, str) or not topics.strip():
        return "Error: Please provide at least one study topic (comma-separated if multiple)."

    try:
        hours = float(available_hours)
    except (TypeError, ValueError):
        return "Error: available_hours must be a valid positive number."

    if hours <= 0:
        return "Error: available_hours must be greater than 0."

    raw_list = []
    for t in topics.replace(";", ",").replace("\n", ",").split(","):
        cleaned = re.sub(r"^(?:and|&|\+)\s+", "", t.strip(), flags=re.IGNORECASE).strip()
        if cleaned:
            raw_list.append(cleaned)
    # Remove duplicates while preserving order
    deduped_topics: list[str] = []
    for item in raw_list:
        if item not in deduped_topics:
            deduped_topics.append(item)

    if not deduped_topics:
        return "Error: Please provide at least one study topic (comma-separated if multiple)."

    num_topics = len(deduped_topics)
    time_per_topic = hours / num_topics

    if time_per_topic >= 1.0:
        time_desc = f"{time_per_topic:.1f} hours"
    else:
        minutes = int(round(time_per_topic * 60))
        time_desc = f"{minutes} minutes"

    plan_lines = [
        f"Structured Study Plan ({hours:.1f} total hours across {num_topics} topic{'s' if num_topics > 1 else ''}):",
    ]
    for idx, topic in enumerate(deduped_topics, start=1):
        plan_lines.append(f"  {idx}. {topic}: ~{time_desc}")
    plan_lines.append("Tip: Use 25-minute Pomodoro intervals with 5-minute active recovery breaks.")

    return "\n".join(plan_lines)


ALL_TOOLS = [days_until_deadline, calculate_study_schedule]
