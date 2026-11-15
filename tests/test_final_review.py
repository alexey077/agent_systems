import json
from pathlib import Path

import pytest

from citizen_agent.cli import select_cases
from citizen_agent.policy import draft_quality_issue

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    s["source_id"]: s["text"] for s in json.loads((ROOT / "data/knowledge.json").read_text())["sources"]
}


@pytest.mark.parametrize(
    "source_id,draft",
    [
        ("fns_tax_notice", "Бумажное письмо не всегда означает отсутствие начислений."),
        ("fns_tax_notice", "Отсутствие бумажного письма не влияет на порядок уплаты налогов."),
        (
            "fns_tax_notice",
            "Отсутствие письма не является основанием для непризнания возможных налоговых обязательств.",
        ),
        ("fns_property_error", "Подайте заявление о перерасчёте налоговой базы."),
    ],
)
def test_observed_final_draft_variants_require_revision(source_id, draft):
    assert draft_quality_issue(draft, [SOURCES[source_id]])


@pytest.mark.parametrize(
    "draft",
    [
        "Отсутствие бумажного письма не означает отсутствия начислений.",
        "Бумажное письмо не пришло, это не означает отсутствия начислений.",
        "Если вы не получили бумажное письмо, это не означает, что начислений нет.",
    ],
)
def test_absence_must_be_checked_in_condition_before_predicate(draft):
    assert draft_quality_issue(draft, [SOURCES["fns_tax_notice"]]) is None


def test_explicitly_supported_tax_base_term_remains_allowed():
    draft = "Сведения о налоговой базе можно уточнить в обращении."
    assert draft_quality_issue(draft, [draft]) is None


def test_multi_case_selection_keeps_order_and_removes_duplicates():
    cases = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    assert select_cases(cases, ["c", "a", "a"]) == [{"id": "a"}, {"id": "c"}]
    assert select_cases(cases, "b") == [{"id": "b"}]
    assert select_cases(cases, None) == cases


def test_unknown_selection_fails_instead_of_silently_changing_sample():
    with pytest.raises(ValueError, match="missing"):
        select_cases([{"id": "a"}], ["a", "missing"])
