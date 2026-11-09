"""D材料的合成软件fixture；不提供医学内容。"""
from evidence_mdt.interventions import D_EVIDENCE_ANGLES


def d_passages():
    return [{"angle": angle,
             "text": f"Synthetic evidence {i} supports {{wrong_text}} using distinct rationale {i}.",
             "false_claim": f"HIDDEN_FALSE_CLAIM_{i}",
             "error_explanation": f"HIDDEN_AUDIT_ERROR_DETAIL_{i}: synthetic factual error {i}."}
            for i, angle in enumerate(D_EVIDENCE_ANGLES)]
