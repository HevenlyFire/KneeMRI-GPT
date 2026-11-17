import json
import os
from urllib import error, request


# Read the DashScope API key from the environment. Never hard-code a real key here.
LLM_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
LLM_API_BASE = os.environ.get(
    "DASHSCOPE_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)

PROBABILITY_THRESHOLD = 0.5


def _extract_message_content(api_result):
    choices = api_result.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    return message.get("content", "")


def _condition_brief(probabilities, threshold=PROBABILITY_THRESHOLD):
    """Rule-based condition brief for fallback recommendations."""
    briefs = []
    if probabilities["acl"] >= threshold:
        briefs.append(
            "ACL tear: injury to the anterior cruciate ligament that may cause "
            "knee instability, swelling, and difficulty with pivoting movements; "
            "severity varies and requires clinical assessment."
        )
    if probabilities["meniscus"] >= threshold:
        briefs.append(
            "Meniscus tear: cartilage injury that may cause joint-line pain, "
            "swelling, catching, or locking; management depends on symptoms and exam."
        )
    if probabilities["abnormal"] >= threshold and not briefs:
        briefs.append(
            "Knee abnormality: imaging suggests a possible structural or soft-tissue "
            "issue; further evaluation by a clinician is recommended."
        )
    if not briefs:
        dominant = max(probabilities, key=probabilities.get)
        return (
            f"No single condition exceeds the threshold ({threshold:.0%}); "
            f"highest probability is {dominant} ({probabilities[dominant]:.1%}). "
            "Maintain routine care and seek evaluation if symptoms develop."
        )
    return " ".join(briefs)


def _self_protection_advice():
    return (
        "Self-protection: avoid high-impact sports and sudden twisting until evaluated; "
        "rest and apply ice for acute pain or swelling; seek prompt medical care if you "
        "cannot bear weight, have severe swelling, or experience locking of the knee. "
        "This guidance does not replace professional diagnosis."
    )


def _default_structured_report(study_id, probabilities, error_message):
    return {
        "study_id": study_id,
        "summary": "LLM call failed; returned a rule-based fallback report.",
        "risk_level": "medium",
        "findings": [
            f"abnormal={probabilities['abnormal']:.4f}",
            f"acl={probabilities['acl']:.4f}",
            f"meniscus={probabilities['meniscus']:.4f}",
        ],
        "recommendations": [
            "Please combine this result with radiologist interpretation for final judgment.",
            _condition_brief(probabilities),
            _self_protection_advice(),
        ],
        "disclaimer": "This report is AI-generated for research and educational reference only, not for final clinical diagnosis.",
        "error": str(error_message),
    }


def _validate_structured_report(structured_report):
    required_fields = [
        "study_id",
        "summary",
        "risk_level",
        "findings",
        "recommendations",
        "disclaimer",
    ]
    for field_name in required_fields:
        if field_name not in structured_report:
            raise ValueError(f"Missing field: {field_name}")

    if structured_report["risk_level"] not in {"low", "medium", "high"}:
        raise ValueError("risk_level is not in the allowed set")

    findings = structured_report["findings"]
    recommendations = structured_report["recommendations"]
    if not isinstance(findings, list) or not isinstance(recommendations, list):
        raise ValueError("findings/recommendations must be arrays")
    if not all(isinstance(item, str) for item in findings):
        raise ValueError("findings must be an array of strings")
    if not all(isinstance(item, str) for item in recommendations):
        raise ValueError("recommendations must be an array of strings")
    if len(recommendations) < 3:
        raise ValueError("recommendations must contain at least 3 items")


def explain_diagnosis_with_qwen(
    study_id,
    abnormal_prob,
    acl_prob,
    meniscus_prob,
    model="qwen3-max",
):
    probabilities = {
        "abnormal": float(abnormal_prob),
        "acl": float(acl_prob),
        "meniscus": float(meniscus_prob),
    }

    system_prompt = (
        "You are a knee MRI assistant for structured report generation. "
        "You must return a strict JSON object and output no text outside JSON. "
        "Required JSON fields: study_id, summary, risk_level, findings, recommendations, disclaimer. "
        "risk_level must be one of low/medium/high. "
        "findings must be an array of strings describing imaging-based probabilities. "
        "recommendations must be an array of at least 3 strings, in this order: "
        "(1) one clinical follow-up or specialist referral suggestion; "
        "(2) one brief patient-friendly explanation of the most relevant specific condition "
        "(abnormal finding, ACL tear, or meniscus tear) that matches the probabilities—"
        "describe what it is and common symptoms without stating a definitive diagnosis; "
        "(3) one self-protection advice for the patient (e.g., avoid high-impact/twisting "
        "activities, rest and ice if acute pain/swelling, seek care if unable to bear weight "
        "or if locking occurs). "
        "Only emphasize conditions with probability >= 0.5, or the single highest probability "
        "if none reach 0.5. Do not discuss low-probability conditions in detail. "
        "All text must be in English."
    )

    user_prompt = {
        "study_id": study_id,
        "task": "knee_mri_structured_report",
        "probabilities": probabilities,
        "constraints": [
            "Do not output any non-JSON text",
            "Do not overstate conclusions; maintain the AI-assistance nature",
            "The disclaimer field is mandatory",
            "recommendations must include: clinical advice, condition brief, and self-protection advice",
            "Condition brief must align with probabilities; use threshold 0.5 or the highest task",
            "Self-protection advice must be practical and remind that this is not a final diagnosis",
        ],
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
        "temperature": 0.2,
    }

    if not LLM_API_KEY:
        return _default_structured_report(
            study_id, probabilities, "DASHSCOPE_API_KEY environment variable is not set"
        )

    endpoint_url = f"{LLM_API_BASE}/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        request_object = request.Request(
            endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with request.urlopen(request_object, timeout=30) as response:
            response_text = response.read().decode("utf-8")
        result = json.loads(response_text)

        raw_content = _extract_message_content(result)
        structured = json.loads(raw_content)
        _validate_structured_report(structured)
        return structured
    except (ValueError, KeyError, json.JSONDecodeError, error.URLError) as exception_object:
        return _default_structured_report(study_id, probabilities, str(exception_object))
