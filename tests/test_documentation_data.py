import json
import re
from pathlib import Path

from citizen_agent.models import Source

ROOT = Path(__file__).resolve().parents[1]


def test_tool_cards_have_all_eight_fields():
    cards = (ROOT / "docs/tool_contracts.md").read_text()
    for name in ("search_knowledge_base", "get_routing_rules", "escalate_to_operator"):
        assert name in cards
    for number in range(1, 9):
        assert cards.count(f"**{number}.") == 3


def test_requirements_are_traceable_and_live_gates_explicit():
    requirements = (ROOT / "docs/requirements.md").read_text()
    for number in range(1, 17):
        assert f"R{number:02}" in requirements
    assert "Финальная запись с реальной моделью" in requirements
    assert "scripted/offline" in requirements


def test_relative_documentation_links_resolve():
    for file in [ROOT / "README.md", *ROOT.glob("docs/*.md")]:
        for target in re.findall(r"\]\(([^)]+)\)", file.read_text()):
            if "://" not in target and not target.startswith("#"):
                assert (file.parent / target.split("#")[0]).exists(), (file, target)


def test_data_have_provenance_and_cover_supported_topics():
    sources = json.loads((ROOT / "data/knowledge.json").read_text())["sources"]
    docs = [Source.model_validate(s) for s in sources]
    assert len({s.source_id for s in docs}) == len(docs)
    assert {s.category for s in docs} == {
        "account_access",
        "tax_notice",
        "property_error",
        "deduction",
        "tax_debt",
        "other",
    }
    assert all(s.checked_at == "2026-11-15" and s.url.startswith("https://www.nalog.gov.ru/") for s in docs)
    routes = json.loads((ROOT / "data/routing.json").read_text())["routes"]
    assert "other" not in routes
    assert all("учебная очередь" in route["recipient"] for route in routes.values())
