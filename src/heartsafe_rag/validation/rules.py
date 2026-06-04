"""HeartSafe VLM-guard rules - cardiology claim validation rules for the AHA/ACC HF Guidelines."""

from vlm_guard import (
    Analysis, BaseRule, RuleResult,
    CrossClaimRule, CrossClaimResult,
)
import re as _re


_CLASS_LOOKUP = {
    "ARNi HFrEF": "Class 1",
    "ACEi HFrEF": "Class 1",
    "ARB HFrEF": "Class 1",
    "Beta blocker HFrEF": "Class 1",
    "MRA HFrEF with LVEF <=35%, NYHA II-IV, on BB+RASi": "Class 1",
    "SGLT2i HFrEF": "Class 1",
    "SGLT2i HFmrEF": "Class 2a",
    "SGLT2i HFpEF": "Class 2a",
    "ARNi HFmrEF": "Class 2a",
    "ARNi HFpEF": "Class 2b",
    "Beta blocker HFmrEF": "Class 2a",
    "MRA HFpEF selected": "Class 2b",
    "Hydralazine+ISDN African American HFrEF NYHA III-IV": "Class 1",
    "Ivabradine HFrEF LVEF <=35% sinus rhythm >=70 on BB": "Class 2a",
    "Digoxin HFrEF": "Class 2b",
    "Tafamidis ATTR-CM NYHA I-III": "Class 1",
    "ICD ischemic LVEF <=30% NYHA I >40d post-MI": "Class 1",
    "ICD non-ischemic LVEF <=35% NYHA II-III GDMT": "Class 1",
    "CRT-D LVEF <=35% LBBB QRS >=150ms NYHA II-III GDMT": "Class 1",
    "CRT-D LVEF <=35% LBBB QRS 120-149ms NYHA II-III GDMT": "Class 2a",
}


_EF_RANGES = [
    (0, 40, "HFrEF"),
    (41, 49, "HFmrEF"),
    (50, 100, "HFpEF"),
]


_GDMT_DRUGS = {
    "RASi": ["ARNi", "ACEi", "ARB"],
    "Beta blocker": ["carvedilol", "metoprolol succinate", "bisoprolol"],
    "MRA": ["spironolactone", "eplerenone"],
    "SGLT2i": ["dapagliflozin", "empagliflozin"],
    "Diuretic": ["furosemide", "torsemide", "bumetanide"],
}


_CONTRANDICATIONS = {
    "MRA": ["K+ >5.0", "Cr >2.5", "eGFR <30", "Addison disease"],
    "ARNi": ["history of angioedema", "K+ >5.0", "systolic BP <100"],
    "Ivabradine": ["sinus node disease", "sick sinus syndrome", "heart block", "resting HR <60", "severe hepatic impairment"],
    "Tafamidis": ["NYHA class IV symptoms"],
}


class CORLevelRule(BaseRule):
    name = "heartsafe.cor_level"
    description = "Validates Class of Recommendation against guideline lookup table"

    def condition(self, analysis: Analysis, context: dict) -> bool:
        return analysis.domain == "cardiology"

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        text = analysis.claim_text.lower()

        cor_mentioned = None
        for cor in ["class 1", "class i", "class 2a", "class iia",
                     "class 2b", "class iib", "class 3", "class iii"]:
            if cor in text:
                cor_mentioned = cor
                break

        if not cor_mentioned:
            analysis.validation_status = "unverifiable"
            return analysis, RuleResult(action_taken=False)

        for key, expected in _CLASS_LOOKUP.items():
            key_lower = key.lower()
            therapy_part = key_lower.split(" ")[0]
            if therapy_part in text:
                expected_lower = expected.lower()
                if cor_mentioned != expected_lower:
                    msg = "Claim states {} for therapy in '{}', but guideline states {}.".format(cor_mentioned.title(), key, expected)
                    analysis.validation_status = "blocked"
                    analysis.validation_message = msg
                    analysis.label = expected
                    return analysis, RuleResult(
                        action_taken=True, action_type="block",
                        message=msg, severity="error",
                        correction_suggestion=msg,
                        modified_fields={"label": expected, "validation_status": "blocked"},
                    )

                analysis.validation_status = "passed"
                analysis.label = expected
                return analysis, RuleResult(
                    action_taken=True, action_type="pass",
                    message="COR {} confirmed for {}".format(expected, key),
                )

        analysis.validation_status = "unverifiable"
        return analysis, RuleResult(action_taken=False)


class LVEFThresholdRule(BaseRule):
    name = "heartsafe.lvef_threshold"
    description = "Validates LVEF ranges and corrects HF classification"

    def condition(self, analysis: Analysis, context: dict) -> bool:
        return analysis.domain == "cardiology" and "lvef" in analysis.claim_text.lower()

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        text = analysis.claim_text.lower()

        ef_match = _re.search(r"lvef\s*(?:is|=|of|:)?\s*(\d+)\s*%?", text)
        if not ef_match:
            ef_match = _re.search(r"(\d+)\s*%\s*(?:lvef|ef|ejection fraction)", text)
        if not ef_match:
            return analysis, RuleResult(action_taken=False)

        ef_value = int(ef_match.group(1))

        correct_label = "Normal EF"
        for lo, hi, label in _EF_RANGES:
            if lo <= ef_value <= hi:
                correct_label = label
                break

        classification_mentioned = analysis.metadata.get("hf_type", analysis.label)
        if classification_mentioned != correct_label and classification_mentioned != "Clinical Finding":
            analysis.label = correct_label
            analysis.validation_status = "corrected"
            msg = "LVEF {}% corresponds to {} (was '{}')".format(ef_value, correct_label, classification_mentioned)
            analysis.validation_message = msg
            return analysis, RuleResult(
                action_taken=True, action_type="correct",
                message=msg, severity="warning",
                correction_suggestion=msg,
                modified_fields={"label": correct_label},
            )

        analysis.validation_status = "passed"
        analysis.label = correct_label
        return analysis, RuleResult(
            action_taken=True, action_type="pass",
            message="LVEF {}% confirmed as {}".format(ef_value, correct_label),
        )


class DrugClassRule(BaseRule):
    name = "heartsafe.drug_class"
    description = "Validates GDMT drug-class relationships"

    def condition(self, analysis: Analysis, context: dict) -> bool:
        return analysis.domain == "cardiology" and analysis.claim_type in ("recommendation", "diagnosis", "contraindication")

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        text = analysis.claim_text.lower()

        for drug_class, drugs in _GDMT_DRUGS.items():
            for drug in drugs:
                if drug.lower() in text:
                    analysis.metadata["drug_class"] = drug_class
                    analysis.metadata["drug_name"] = drug
                    return analysis, RuleResult(
                        action_taken=True, action_type="pass",
                        message="Drug '{}' recognized in class '{}'".format(drug, drug_class),
                    )

        return analysis, RuleResult(action_taken=False)


class ContraindicationRule(BaseRule):
    name = "heartsafe.contraindication"
    description = "Flags contraindications for GDMT therapies"

    def condition(self, analysis: Analysis, context: dict) -> bool:
        return analysis.domain == "cardiology"

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        text = analysis.claim_text.lower()

        for therapy, contraindications in _CONTRANDICATIONS.items():
            if therapy.lower() in text:
                flags = []
                for ci in contraindications:
                    if ci.lower() in text:
                        flags.append(ci)

                if flags:
                    msg = "Contraindication flag for {}: {}".format(therapy, ", ".join(flags))
                    analysis.validation_status = "flagged"
                    analysis.validation_message = msg
                    analysis.recommendation = (
                        "Caution: {} is contraindicated when: {}. "
                        "Verify patient values before recommending."
                    ).format(therapy, ", ".join(contraindications))
                    return analysis, RuleResult(
                        action_taken=True, action_type="flag",
                        message=msg, severity="warning",
                        modified_fields={"validation_status": "flagged", "recommendation": analysis.recommendation},
                    )

        return analysis, RuleResult(action_taken=False)


class ValueStatementRule(BaseRule):
    name = "heartsafe.value_statement"
    description = "Validates value statements (high/low value)"

    def condition(self, analysis: Analysis, context: dict) -> bool:
        return analysis.domain == "cardiology" and "value" in analysis.claim_text.lower()

    def action(self, analysis: Analysis, context: dict) -> tuple[Analysis, RuleResult]:
        text = analysis.claim_text.lower()

        if "high value" in text:
            if "$60,000" not in text and "60000" not in text:
                msg = "High value is defined as <$60,000 per QALY gained. Please include the threshold."
                analysis.validation_status = "flagged"
                analysis.validation_message = msg
                return analysis, RuleResult(
                    action_taken=True, action_type="flag",
                    message=msg, severity="info",
                )

            if "tafamidis" in text or "cardiac amyloidosis" in text:
                msg = "Tafamidis was identified as low value in the 2022 guideline."
                analysis.validation_status = "flagged"
                analysis.validation_message = msg
                return analysis, RuleResult(
                    action_taken=True, action_type="flag",
                    message=msg, severity="info",
                )

            analysis.validation_status = "passed"
            return analysis, RuleResult(
                action_taken=True, action_type="pass",
                message="High value statement confirmed",
            )

        if "low value" in text:
            if "tafamidis" not in text:
                msg = "The only therapy identified as low value was tafamidis for cardiac amyloidosis."
                analysis.validation_status = "corrected"
                analysis.validation_message = msg
                return analysis, RuleResult(
                    action_taken=True, action_type="correct",
                    message=msg, severity="warning",
                    correction_suggestion=msg,
                )

            analysis.validation_status = "passed"
            return analysis, RuleResult(
                action_taken=True, action_type="pass",
                message="Low value statement confirmed",
            )

        return analysis, RuleResult(action_taken=False)


class AnswerConsistencyRule(CrossClaimRule):
    name = "heartsafe.answer_consistency"
    description = "Cross-validates final answer against all validated claims"
    order = 2000

    def condition(self, claims: list[Analysis], answer: str, context: dict) -> bool:
        return bool(claims) and bool(answer)

    def action(self, claims: list[Analysis], answer: str, context: dict) -> tuple[list[Analysis], str, CrossClaimResult]:
        answer_lower = answer.lower()
        neg_verbs = ["not recommended", "contraindicated", "should not", "avoid", "do not use"]
        has_negation = any(v in answer_lower for v in neg_verbs)

        has_class1_warning = False
        for c in claims:
            if c.validation_status in ("passed", "corrected", "unverifiable"):
                label_lower = c.label.lower()
                claim_lower = c.claim_text.lower()
                if ("class 1" in label_lower or "class i" in label_lower or "class 1" in claim_lower) and has_negation:
                    has_class1_warning = True
                    break

        if has_class1_warning:
            corrected = answer
            for verb in neg_verbs:
                if verb in answer_lower:
                    corrected = answer.replace(verb, "[VERIFY CLAIMS] " + verb)
                    break
            return claims, corrected, CrossClaimResult(
                action_taken=True, action_type="block",
                message="Answer contradicts Class 1 recommendation in claims.",
                severity="error",
                modified_answer=corrected,
                correction_suggestion="Answer says 'not recommended' but claims indicate Class 1 recommendation.",
            )

        has_flag = any(c.validation_status == "flagged" for c in claims)
        if has_flag:
            return claims, answer, CrossClaimResult(
                action_taken=True, action_type="flag",
                message="Some claims were flagged - review before accepting.",
                severity="warning",
            )

        return claims, answer, CrossClaimResult(action_taken=False)
