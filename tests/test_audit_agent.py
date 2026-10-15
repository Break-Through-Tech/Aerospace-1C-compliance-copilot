"""Behavioral boundaries for the agent against the real validated pipeline."""

import copy
import json

import pytest
from pydantic import ValidationError

from compliance_copilot.agent import AgentResult, AuditAgent
from compliance_copilot.pipeline import AuditPipeline
from compliance_copilot.schemas import Clause


REQUIREMENT = "The project shall document a software test plan."
CLAUSE = {
    "swe_id": "SWE-001", "section": "3.1.1",
    "requirement_text": "The project shall document and review a software test plan.",
    "source_url": "https://example.org/nasa", "text_scope": "reviewed_full_text",
    "rank": 1, "score": 0.8,
}


def call(tool, **arguments):
    return json.dumps({"tool": tool, "arguments": arguments})


def sequence(verdict="Partial"):
    actions = [call("retrieve_clause", query=REQUIREMENT), call("check_requirement")]
    if verdict != "Meets":
        actions.append(call("log_gap"))
    return actions + [call("finish")]


class ScriptedController:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.prompts = []

    def __call__(self, messages):
        self.prompts.append(copy.deepcopy(messages))
        action = next(self.actions)
        if isinstance(action, Exception):
            raise action
        return action


def harness(actions, *, verdict="Partial", retrieved=None, generated=None, max_steps=8):
    retrieval_calls, generation_prompts = [], []

    def retrieve(query, top_k):
        retrieval_calls.append((query, top_k))
        if callable(retrieved):
            return retrieved(query)
        return [CLAUSE] if retrieved is None else retrieved

    def generate(messages):
        generation_prompts.append(copy.deepcopy(messages))
        if isinstance(generated, Exception):
            raise generated
        if generated is not None:
            return generated
        return json.dumps({"verdict": verdict, "reasoning": "Review is not required by the proposal.",
                           "cited_swe_ids": ["SWE-001"]})

    controller = ScriptedController(actions)
    agent = AuditAgent(AuditPipeline(retrieve, generate, top_k=3), controller, max_steps=max_steps)
    return agent, controller, retrieval_calls, generation_prompts


@pytest.mark.parametrize("verdict", ["Meets", "Partial", "Gap"])
def test_completed_audits_preserve_pipeline_citations_and_trace(verdict):
    agent, controller, retrieval_calls, generation_prompts = harness(sequence(verdict), verdict=verdict)
    result = agent.audit("REQ-private-label", REQUIREMENT)
    assert result.termination == "finished"
    assert result.result.status == "ok"
    assert result.result.audit.verdict == verdict
    assert retrieval_calls == [(REQUIREMENT, 3)]
    assert len(generation_prompts) == 1
    assert [step.tool for step in result.steps] == [json.loads(a)["tool"] for a in sequence(verdict)]
    assert [step.step for step in result.steps] == list(range(1, len(result.steps) + 1))
    citation = result.result.audit.citations[0]
    assert citation.model_dump() == {"swe_id": CLAUSE["swe_id"], "section": CLAUSE["section"],
                                     "quote": CLAUSE["requirement_text"]}
    assert len(result.gaps) == (0 if verdict == "Meets" else 1)
    if result.gaps:
        assert result.gaps[0].audit == result.result.audit
        assert result.gaps[0].requirement_text == REQUIREMENT
    assert AgentResult.model_validate_json(result.model_dump_json()) == result
    # Tool observations include the source records needed for an informed next action.
    observation = json.loads(controller.prompts[1][-1]["content"])
    assert observation["observation"]["clauses"][0] == CLAUSE


def test_refined_query_only_changes_retrieval_not_original_requirement():
    actions = [call("retrieve_clause", query=REQUIREMENT),
               call("retrieve_clause", query="software verification planning"),
               call("check_requirement"), call("finish")]
    agent, _, retrieval_calls, prompts = harness(actions, verdict="Meets")
    result = agent.audit("REQ-1", REQUIREMENT)
    assert result.termination == "finished"
    assert retrieval_calls == [(REQUIREMENT, 3), ("software verification planning", 3)]
    assert json.loads(prompts[0][1]["content"])["requirement"] == REQUIREMENT
    assert result.result.requirement_text == REQUIREMENT


@pytest.mark.parametrize("raw", [
    "not json", "```json\n{}\n```", None, {},
    call("unknown"), call("retrieve_clause", query=""),
    call("retrieve_clause", query=REQUIREMENT, top_k=100),
    call("check_requirement", requirement="Rewrite the original"),
    call("log_gap", verdict="Meets"), call("finish", result="ok"),
    json.dumps({"tool": "finish", "arguments": {}, "audit": {}}),
])
def test_invalid_tool_calls_fail_closed_without_touching_pipeline(raw):
    agent, _, retrieval_calls, prompts = harness([raw])
    result = agent.audit("REQ-1", REQUIREMENT)
    assert result.termination == "invalid_tool_call"
    assert result.result.status == "error"
    assert not result.gaps and not retrieval_calls and not prompts


@pytest.mark.parametrize("actions, verdict, code, retrieval_count, generation_count", [
    ([call("check_requirement")], "Partial", "invalid_state", 0, 0),
    ([call("log_gap")], "Partial", "invalid_state", 0, 0),
    ([call("finish")], "Partial", "incomplete_audit", 0, 0),
    ([call("retrieve_clause", query=REQUIREMENT), call("finish")], "Partial", "incomplete_audit", 1, 0),
    ([call("retrieve_clause", query=REQUIREMENT), call("log_gap")], "Partial", "invalid_state", 1, 0),
    ([call("retrieve_clause", query=REQUIREMENT), call("check_requirement"), call("finish")],
     "Partial", "incomplete_audit", 1, 1),
    ([call("retrieve_clause", query=REQUIREMENT), call("check_requirement"), call("log_gap")],
     "Meets", "invalid_state", 1, 1),
    ([call("retrieve_clause", query=REQUIREMENT), call("check_requirement"), call("check_requirement")],
     "Meets", "invalid_state", 1, 1),
    ([call("retrieve_clause", query=REQUIREMENT), call("check_requirement"),
      call("retrieve_clause", query="replace evidence")], "Meets", "already_checked", 1, 1),
])
def test_tool_ordering_rejects_incomplete_or_mutated_audits(
        actions, verdict, code, retrieval_count, generation_count):
    agent, _, retrieval_calls, prompts = harness(actions, verdict=verdict)
    result = agent.audit("REQ-1", REQUIREMENT)
    assert result.termination == "tool_error"
    assert result.result.audit is None and result.result.status == "error"
    assert result.steps[-1].observation == {"status": "error", "code": code}
    assert len(result.steps) == len(actions)
    assert not result.gaps
    assert len(prompts) == generation_count
    assert len(retrieval_calls) == retrieval_count


def test_duplicate_gap_log_is_idempotent():
    actions = sequence()[:-1] + [call("log_gap"), call("finish")]
    agent, _, _, prompts = harness(actions)
    result = agent.audit("REQ-1", REQUIREMENT)
    assert result.termination == "finished"
    assert len(result.gaps) == len(prompts) == 1
    assert result.steps[2].observation == result.steps[3].observation


def test_empty_retrieval_can_be_requeried():
    actions = [call("retrieve_clause"), call("retrieve_clause", query="refined search"),
               call("check_requirement"), call("finish")]
    agent, _, calls, prompts = harness(actions, verdict="Meets",
                                       retrieved=lambda query: [] if query == REQUIREMENT else [CLAUSE])
    result = agent.audit("REQ-1", REQUIREMENT)
    assert result.termination == "finished"
    assert len(calls) == 2 and len(prompts) == 1


def test_empty_latest_retrieval_cannot_reuse_prior_context():
    actions = [call("retrieve_clause", query=REQUIREMENT), call("retrieve_clause", query="no match"),
               call("check_requirement")]
    agent, _, _, prompts = harness(actions, retrieved=lambda query: [] if query == "no match" else [CLAUSE])
    result = agent.audit("REQ-1", REQUIREMENT)
    assert result.termination == "tool_error"
    assert not prompts and not result.result.retrieved


def test_retriever_error_discards_previous_context():
    def retrieve(query):
        if query == "broken":
            raise RuntimeError("private backend details")
        return [CLAUSE]
    agent, _, _, prompts = harness([call("retrieve_clause", query=REQUIREMENT),
                                   call("retrieve_clause", query="broken")], retrieved=retrieve)
    result = agent.audit("REQ-1", REQUIREMENT)
    assert result.termination == "tool_error"
    assert result.result.error_code == "retrieval_error"
    assert not result.result.retrieved and not prompts
    assert "private backend" not in result.model_dump_json()


@pytest.mark.parametrize("generated, code", [
    ("not json", "invalid_output"),
    (json.dumps({"verdict": "Meets", "reasoning": "Invented source", "cited_swe_ids": ["SWE-999"]}), "invalid_citation"),
    (json.dumps({"error": "no_relevant_clause"}), "no_context"),
    (RuntimeError("private generation details"), "generation_error"),
])
def test_pipeline_failure_codes_survive_agent_boundary(generated, code):
    agent, _, _, _ = harness(sequence(), generated=generated)
    result = agent.audit("REQ-1", REQUIREMENT)
    assert result.termination == "audit_error"
    assert result.result.error_code == code
    assert not result.gaps and result.result.audit is None
    assert len(result.steps) == 2


def test_controller_exception_is_safe_and_does_not_commit_staged_gap():
    agent, _, _, _ = harness(sequence()[:-1] + [RuntimeError("private credentials")])
    result = agent.audit("REQ-1", REQUIREMENT)
    assert result.termination == "controller_error"
    assert result.result.error_code == "generation_error"
    assert result.result.audit is None and not result.gaps
    assert "private credentials" not in result.model_dump_json()


@pytest.mark.parametrize("actions, limit", [
    ([call("retrieve_clause", query=REQUIREMENT)] * 8, 2),
    (sequence(), 3),  # A logged gap is still provisional until finish succeeds.
])
def test_step_limit_never_exports_success(actions, limit):
    agent, controller, _, _ = harness(actions, max_steps=limit)
    result = agent.audit("REQ-1", REQUIREMENT)
    assert result.termination == "step_limit"
    assert len(result.steps) == len(controller.prompts) == limit
    assert not result.gaps and result.result.audit is None
    assert AgentResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("limit", [0, -1, 33, True, 1.5, "8"])
def test_invalid_step_budget_is_rejected(limit):
    with pytest.raises(ValueError):
        harness([], max_steps=limit)


def test_runs_do_not_share_context_or_staged_gaps():
    agent, controller, retrieval_calls, prompts = harness(sequence() + [call("finish")])
    first = agent.audit("REQ-1", REQUIREMENT)
    second = agent.audit("REQ-2", "A different requirement")
    assert first.termination == "finished"
    assert second.termination == "tool_error"
    assert not second.result.retrieved and not second.gaps
    assert second.result.requirement_id == "REQ-2"
    assert len(second.steps) == 1 and len(controller.prompts[-1]) == 2
    assert len(retrieval_calls) == len(prompts) == 1


def test_initial_controller_prompt_excludes_benchmark_identifier():
    identifier = "benchmark-Gap-expected-SWE-999"
    agent, controller, _, _ = harness([call("finish")])
    agent.audit(identifier, REQUIREMENT)
    initial = controller.prompts[0]
    assert identifier not in json.dumps(initial)
    payload = json.loads(initial[1]["content"])
    assert set(payload) == {"requirement", "tool_call_schema", "max_steps"}
    assert payload["requirement"] == REQUIREMENT


@pytest.mark.parametrize("mutation, error", [
    ("failed_success", "Only finished agent runs may export successful audits"),
    ("unlogged", "Finished runs must log every Partial or Gap audit"),
    ("wrong_requirement", "Gap must match the completed audit exactly"),
    ("wrong_audit", "Gap must match the completed audit exactly"),
    ("extra_gap", "List should have at most 1 item"),
])
def test_result_model_rejects_impossible_completed_records(mutation, error):
    agent, _, _, _ = harness(sequence())
    result = agent.audit("REQ-1", REQUIREMENT)
    data = result.model_dump()
    assert AgentResult.model_validate(data) == result
    if mutation == "failed_success":
        data["termination"] = "step_limit"
    elif mutation == "unlogged":
        data["gaps"] = []
    elif mutation == "wrong_requirement":
        data["gaps"][0]["requirement_text"] = "An unrelated proposal"
    elif mutation == "wrong_audit":
        data["gaps"][0]["audit"]["reasoning"] = "Changed reasoning"
    else:
        data["gaps"].append(copy.deepcopy(data["gaps"][0]))
    with pytest.raises(ValidationError, match=error):
        AgentResult.model_validate(data)


def test_first_retrieval_cannot_rewrite_original_requirement():
    agent, _, retrieval_calls, prompts = harness([call("retrieve_clause", query="test plan")])
    result = agent.audit("REQ-1", REQUIREMENT)
    assert result.termination == "tool_error"
    assert result.steps[0].observation["code"] == "initial_query_changed"
    assert not retrieval_calls and not prompts
    assert result.result.audit is None and not result.gaps


@pytest.mark.parametrize("arguments", [{}, {"query": None}])
def test_first_retrieval_defaults_to_original_requirement(arguments):
    actions = [call("retrieve_clause", **arguments), call("check_requirement"), call("finish")]
    agent, _, retrieval_calls, prompts = harness(actions, verdict="Meets")
    result = agent.audit("REQ-1", REQUIREMENT)
    assert result.termination == "finished"
    assert retrieval_calls == [(REQUIREMENT, 3)]
    assert result.steps[0].observation["query"] == REQUIREMENT
    assert json.loads(prompts[0][1]["content"])["requirement"] == REQUIREMENT


@pytest.mark.parametrize("as_model", [False, True])
def test_external_source_mutation_cannot_rewrite_evidence_after_retrieval(as_model):
    source = Clause.model_validate(CLAUSE) if as_model else copy.deepcopy(CLAUSE)
    agent, scripted, _, prompts = harness(sequence(), retrieved=[source])

    def controller(messages):
        if len(scripted.prompts) == 1:
            if as_model:
                source.requirement_text = "Altered after retrieval."
                source.section = "9.9.9"
            else:
                source["requirement_text"] = "Altered after retrieval."
                source["section"] = "9.9.9"
        return scripted(messages)

    agent.controller = controller
    result = agent.audit("REQ-1", REQUIREMENT)
    assert result.termination == "finished"
    reference = json.loads(prompts[0][1]["content"])["references"][0]
    assert reference["text"] == CLAUSE["requirement_text"]
    assert reference["section"] == CLAUSE["section"]
    assert result.gaps[0].audit.citations[0].quote == CLAUSE["requirement_text"]
    assert result.gaps[0].audit.citations[0].section == CLAUSE["section"]


@pytest.mark.parametrize("verdict", ["Partial", "Meets"])
def test_controller_receives_only_current_state_tool_schema(verdict):
    actions = [call("retrieve_clause"), call("retrieve_clause", query="refined search"),
               call("check_requirement")]
    if verdict == "Partial":
        actions.append(call("log_gap"))
    actions.append(call("finish"))
    agent, controller, _, _ = harness(
        actions, verdict=verdict,
        retrieved=lambda query: [] if query == REQUIREMENT else [CLAUSE],
    )
    assert agent.audit("REQ-1", REQUIREMENT).termination == "finished"
    schemas = [json.loads(prompt[-1]["content"])["tool_call_schema"]
               for prompt in controller.prompts]

    assert schemas[0]["title"] == "RetrieveOriginalCall"
    original_arguments = schemas[0]["$defs"]["EmptyArguments"]
    assert original_arguments["properties"] == {}
    assert original_arguments["additionalProperties"] is False
    assert schemas[1]["title"] == "RetrieveCall"
    assert "query" in schemas[1]["$defs"]["RetrievalArguments"]["properties"]
    assert {choice["$ref"] for choice in schemas[2]["anyOf"]} == {
        "#/$defs/RetrieveCall", "#/$defs/CheckCall",
    }
    assert [schema["title"] for schema in schemas[3:]] == (
        ["LogCall", "FinishCall"] if verdict == "Partial" else ["FinishCall"]
    )
