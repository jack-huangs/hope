"""集中放置提示词：先说明谁来做什么，最后说明 JSON 格式。"""
from __future__ import annotations

import json
from typing import Any

PROMPT_VERSION = "specialist-responsibilities-v2.10-medxpertqa-bcd-history-citations"

RECRUIT = """You coordinate a four-person medical panel for the supplied question.
Select exactly four relevant medical specialties with complementary responsibilities.
For each expert, provide a role, a short description of expertise, and the
question-relevant aspect they should assess. Assign assessment responsibilities,
not preferred answers. Base the selection only on the supplied question and options.
Do not solve the question or insert a diagnosis or treatment recommendation into the role description.
Return exactly one JSON object: {"experts":[{"role":"...","expertise":"...","focus":"..."}]}.
Return exactly four experts with distinct roles and nonempty string fields."""

INITIAL = """This is your independent initial assessment. No peer assessment is available.
Identify the case features relevant to your assigned responsibility. From your specialist perspective:
1. Explain the disease or clinical condition suggested by the case.
2. Identify the key symptoms, signs, and medical clues relevant to the assessment.
3. Highlight the most important abnormalities and clinical risks that could affect the decision.
4. Provide a specialty-specific analysis of the case, explaining how the relevant findings support or weaken the plausible interpretations.
Do not treat missing information as established fact.
State the claims you rely on, relevant counterarguments.
Focus on decision-relevant evidence rather than listing or summarizing every supplied passage.
Provide your answer, probability distribution, and a concise specialist summary using the required JSON schema."""

DISCUSSION = """This is discussion round {round_id}. Continue acting as the {role}.
Your latest previous assessment is supplied as self_previous. All other earlier assessments,
including your own older assessments, are supplied in messages with role and round information.
Review the full available discussion history and distinguish earlier from latest judgments.
These are discussion inputs for you to evaluate.
Review the clinically important points that affect your assigned responsibility.
Explain your assessment of the most consequential agreement or disagreement,
including the case finding or supplied evidence that bears on it.
Then state your current judgment and whether it changed from your previous one.
If it changed, identify the specific information or argument responsible.
If it did not change, briefly explain why. Agreement and disagreement are both acceptable;
do not invent a disagreement or a change to satisfy this task.
Your claims should express the reasons you currently endorse, dispute, or qualify.
You are contributing a specialist assessment, not writing the whole panel's report.
Use opinion for the concise update explanation."""

REVIEW = """Continue as the {role}. Additional evidence is supplied in additional_evidence,
separately from the previously available evidence in evidence.
Your previous assessment is supplied as self_previous; messages contains peer assessments.
Assess whether the additional evidence changes any clinically important claim within your responsibility,
and explain the applicable conditions or limits. Update your judgment only as justified by your assessment.
If the added evidence does not resolve the important uncertainty, state what remains unresolved.
Use opinion to explain whether and why your judgment changed. Return the same expert-response JSON schema."""

HOST = """You are the clinical moderator making the final decision for this question.
Integrate the specialists' assessments with the supplied case and evidence.
Review all expert assessments from R0 onward; distinguish earlier from latest judgments by role and round.
Additional evidence, when present, is supplied separately in additional_evidence.
Resolve the clinically important differences where the available information allows it,
and state any remaining uncertainty. Give the best-supported option and your probability distribution
using the required JSON schema. Set unexplained_contradiction to true when a material conflict
affecting the decision remains unresolved. Set missing_distinction to true when available information
does not resolve a distinction needed to choose between plausible options.
These flags concern decision-relevant gaps, not every omitted detail."""


def response_system(payload: dict[str, Any]) -> str:
    """修复调用沿用原阶段身份，但只能修格式，不能重做判断。"""
    repair = payload["task"] == "repair"
    original = payload.get("original_request", payload) if repair else payload
    task = original["task"]
    role = original.get("role", "medical expert")
    if task == "host":
        stage = HOST
    else:
        stage = (f"You are the {role} on a multidisciplinary medical panel.\n"
                 f"Your expertise is {original.get('expertise', 'medical assessment')}. "
                 f"Your responsibility in this case is {original.get('focus', 'assess the case')}. "
                 "Your role should shape what you examine and how you assess it.\n"
                 "Consider relevant findings outside your specialty when they affect the decision.\n"
                 "If your specialty has little to add, say so; do not invent a specialist finding.\n"
                 "State your current best answer.\n")
        stage += (REVIEW.format(role=role) if task == "review" else
                  INITIAL if original.get("round_id", 0) == 0 else
                  DISCUSSION.format(role=role, round_id=original["round_id"]))
    labels = list(original["question"]["options"])
    # opinion 允许说明更新；独立的医学理由仍需进入 claims，便于语义比较。
    result = stage + "\n" + (
        "Evidence is provided for your reference. "
        "Return exactly one JSON object. This is a research response, not clinical advice. "
        f"The only valid answer and claims[].answer values are option KEYS {json.dumps(labels)}; "
        "never put option text there. "
        "Before returning your response, verify that your answer key and the option text are mutually consistent. "
        f"probabilities must have exactly these keys and numeric values: {json.dumps({k: 'number in [0,1]' for k in labels})}; "
        "every value must be between 0 and 1. "
        "Use the IDs attached to the supplied evidence and discussion history when citing them. "
        "Do not invent IDs. "
        "Return answer, probabilities, citations (array of IDs), claims (array of objects with "
        "text, answer, citation_ids, polarity, conditions), and opinion (short string). "
        "Each claim polarity must be support, contradict, limit, or uncertain; "
        "conditions must be one STRING (use an empty string if none), never an array. "
        "Put each distinct substantive medical reason, limitation, and counterevidence in claims. "
        "opinion is a concise specialist summary and may explain whether and why your judgment changed; "
        "do not leave a substantive medical reason only in opinion. "
        "Claim attributions show the speaker role and the round in which the claim was expressed. "
        "For host responses also return unexplained_contradiction and missing_distinction as booleans. "
        "Do not use a gold label or assume an option is correct from its key."
    )
    if repair:
        result += (" Correct only the structure of original_response using validation_error. "
                   "Preserve its substantive judgment and every original numeric probability exactly. "
                   "Map option text to its unique key if needed; never change the selected option. "
                   "Do not add new medical reasoning or calculate a new probability vector. "
                   "If the response cannot be faithfully repaired, return an empty JSON object. "
                   "Return no commentary or wrapper.")
    return result
