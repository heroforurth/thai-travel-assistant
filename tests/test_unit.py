"""Unit tests: no proxy needed (the model is stubbed). Run: pytest -q"""
import json

import pytest
from pydantic import ValidationError

from app import agent
from app.chunking import Chunk, chunk_guide, chunk_text
from app.retrieve import Hit
from app.schemas import parse_action
from app.tools import calculate
from eval.common import is_abstain, keyword_correct


# --- sandboxed calculator -------------------------------------------------------------

def test_calculator_arithmetic():
    assert calculate("750 * 2 * 2") == 3000
    assert calculate("(1600 - 600) * 5") == 5000
    assert calculate("10 / 4") == 2.5


@pytest.mark.parametrize("expr", ["__import__('os').system('dir')", "open('x')", "a + 1", "2 ** 1000", "[1, 2]"])
def test_calculator_rejects_non_arithmetic(expr):
    with pytest.raises((ValueError, SyntaxError)):
        calculate(expr)


# --- structured output validation -----------------------------------------------------

def test_parse_search_with_fences_and_case_insensitive_destination():
    a = parse_action('```json\n{"action": "search", "query": "ferry", "destination": "ko tao"}\n```')
    assert a.action == "search" and a.destination == "Ko Tao"


def test_unknown_destination_is_rejected():
    with pytest.raises(ValidationError):
        parse_action('{"action": "search", "query": "ferry", "destination": "Atlantis"}')


def test_calculate_only_accepts_arithmetic_characters():
    with pytest.raises(ValidationError):
        parse_action('{"action": "calculate", "expression": "import os"}')


def test_unknown_action_is_rejected():
    with pytest.raises(ValidationError):
        parse_action('{"action": "delete_files"}')


def test_no_json_is_rejected():
    with pytest.raises(ValueError):
        parse_action("Sure! The ferry costs 750 baht.")


# --- chunking ---------------------------------------------------------------------------

def test_chunks_respect_size_and_keep_sections():
    text = "Intro line.\n\n== Get in ==\n" + "\n".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_guide("Pai", text, size=300, overlap=50)
    assert chunks[0].section == "Overview"
    assert {c.section for c in chunks[1:]} == {"Get in"}
    assert all(len(c.text) <= 300 + 50 for c in chunks)
    assert chunks[1].id == "pai/get-in/0"


def test_overlap_carries_tail_forward():
    parts = chunk_text("aaaa bbbb cccc\ndddd eeee ffff\ngggg hhhh iiii", size=20, overlap=9)
    assert len(parts) > 1 and parts[1].startswith("cccc")


# --- the agent loop (model stubbed) -----------------------------------------------------

class FakeIndex:
    def __init__(self):
        self.chunk = Chunk("ko-tao/get-in/2", "Ko Tao", "Get in", "The fare for a Lomprayah ferry is 750 baht.")

    def search(self, query, **kw):
        return [Hit(self.chunk, 0.9, 0.9)]


def script(monkeypatch, replies):
    it = iter(replies)
    monkeypatch.setattr(agent.llm, "chat", lambda messages, **kw: next(it))


def test_agent_search_calculate_final(monkeypatch):
    script(monkeypatch, [
        json.dumps({"action": "search", "query": "Lomprayah fare", "destination": "Ko Tao"}),
        json.dumps({"action": "calculate", "expression": "750 * 4"}),
        json.dumps({"action": "final", "answer": "3,000 baht.", "citations": ["ko-tao/get-in/2", "made-up/0"]}),
    ])
    res = agent.run("Two people round trip?", FakeIndex())
    assert res.stopped_by == "final" and res.steps == 3
    assert [c.id for c in res.citations] == ["ko-tao/get-in/2"]
    assert res.dropped_citations == ["made-up/0"]  # never retrieved -> not shown as a source
    assert "750 * 4 = 3000" in res.trace[1]["observation"]


def test_agent_repairs_one_bad_step(monkeypatch):
    script(monkeypatch, [
        "I think it is 750 baht",  # invalid -> repair prompt
        json.dumps({"action": "final", "answer": agent.ABSTAIN, "citations": []}),
    ])
    res = agent.run("fare?", FakeIndex())
    assert res.parse_failures == 1 and res.stopped_by == "final"


def test_agent_stops_at_max_steps(monkeypatch):
    search = json.dumps({"action": "search", "query": "ferry", "destination": None})
    script(monkeypatch, [search] * 10)
    res = agent.run("fare?", FakeIndex(), max_steps=3)
    assert res.stopped_by == "max_steps" and res.answer == agent.ABSTAIN


# --- eval scoring rules -----------------------------------------------------------------

def test_keyword_scoring():
    ex = {"type": "calc", "keywords": ["3,000"]}
    assert keyword_correct("They pay 3000 baht in total.", ex)
    assert not keyword_correct("They pay 1500 baht.", ex)
    una = {"type": "unanswerable", "keywords": []}
    assert keyword_correct("I couldn't find this in the travel guides.", una)
    assert not keyword_correct("Ko Lipe is reached by ferry from Pak Bara.", una)
    assert is_abstain("The guides do not mention a metro.")
