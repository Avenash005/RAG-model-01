"""Unit tests for custom academic tools."""

import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import days_until_deadline, calculate_study_schedule


def test_days_until_deadline_future():
    # Target 10 days in the future
    future_date = (datetime.date.today() + datetime.timedelta(days=10)).strftime("%Y-%m-%d")
    result = days_until_deadline.invoke({"deadline": future_date})
    assert "10 days remaining" in result


def test_days_until_deadline_tomorrow():
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    result = days_until_deadline.invoke({"deadline": tomorrow})
    assert "Tomorrow, 1 day remaining" in result


def test_days_until_deadline_today():
    today = datetime.date.today().strftime("%Y-%m-%d")
    result = days_until_deadline.invoke({"deadline": today})
    assert "is TODAY" in result


def test_days_until_deadline_past():
    past_date = (datetime.date.today() - datetime.timedelta(days=5)).strftime("%Y-%m-%d")
    result = days_until_deadline.invoke({"deadline": past_date})
    assert "passed 5 days ago" in result


def test_days_until_deadline_invalid():
    result = days_until_deadline.invoke({"deadline": "not-a-valid-date-str"})
    assert result.startswith("Error: Unable to parse deadline")


def test_days_until_deadline_empty():
    result = days_until_deadline.invoke({"deadline": "   "})
    assert "Error:" in result


def test_study_schedule_valid():
    result = calculate_study_schedule.invoke({
        "topics": "Machine Learning, Neural Networks, RAG",
        "available_hours": 6.0,
    })
    assert "Structured Study Plan (6.0 total hours across 3 topics)" in result
    assert "Machine Learning: ~2.0 hours" in result
    assert "Neural Networks: ~2.0 hours" in result
    assert "RAG: ~2.0 hours" in result
    assert "Pomodoro" in result


def test_study_schedule_short_duration_minutes():
    result = calculate_study_schedule.invoke({
        "topics": "Math, Physics",
        "available_hours": 1.0,
    })
    assert "30 minutes" in result


def test_study_schedule_invalid_hours():
    result = calculate_study_schedule.invoke({
        "topics": "Math",
        "available_hours": -2.0,
    })
    assert "Error: available_hours must be greater than 0" in result


def test_study_schedule_empty_topics():
    result = calculate_study_schedule.invoke({
        "topics": " , , ",
        "available_hours": 5.0,
    })
    assert "Error: Please provide at least one study topic" in result


if __name__ == "__main__":
    test_days_until_deadline_future()
    test_days_until_deadline_tomorrow()
    test_days_until_deadline_today()
    test_days_until_deadline_past()
    test_days_until_deadline_invalid()
    test_days_until_deadline_empty()
    test_study_schedule_valid()
    test_study_schedule_short_duration_minutes()
    test_study_schedule_invalid_hours()
    test_study_schedule_empty_topics()
    print("All tool tests passed successfully!")
