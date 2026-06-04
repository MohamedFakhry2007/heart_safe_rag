"""Test implicit COR detection and CSV-based rules."""

from vlm_guard import Analysis

from heartsafe_rag.validation.rules import CORLevelRule, CSVCorRule


def test_implicit_cor_no_benefit() -> None:
    rule = CORLevelRule()
    a = Analysis(label="", claim_text="Anticoagulation provides no benefit in HF patients in sinus rhythm.", domain="cardiology", claim_type="recommendation")  # fmt: skip
    _, r = rule.action(a, {})
    assert a.validation_status in ("passed", "blocked"), "Should have matched implicit 'no benefit'"
    assert r.action_taken


def test_implicit_cor_may_be_considered() -> None:
    rule = CORLevelRule()
    a = Analysis(label="", claim_text="Beta blocker therapy may be considered in patients with HFmrEF.", domain="cardiology", claim_type="recommendation")  # fmt: skip
    _, r = rule.action(a, {})
    assert a.validation_status in ("passed", "blocked"), "Should have matched implicit 'may be considered'"
    assert r.action_taken


def test_implicit_cor_is_recommended() -> None:
    rule = CORLevelRule()
    a = Analysis(label="", claim_text="Beta blocker therapy is recommended in patients with HFrEF.", domain="cardiology", claim_type="recommendation")  # fmt: skip
    _, r = rule.action(a, {})
    assert a.validation_status == "passed", "'is recommended' Class 1 should match Beta blocker HFrEF Class 1"
    assert r.action_taken


def test_csv_anticoagulation_not_recommended() -> None:
    csv = CSVCorRule()
    a = Analysis(label="", claim_text="Anticoagulation is not recommended for HF patients in sinus rhythm.", domain="cardiology", claim_type="recommendation")  # fmt: skip
    _, r = csv.action(a, {})
    assert a.validation_status in ("passed", "blocked"), "Should have matched CSV anticoagulation rule"
    assert r.action_taken


def test_csv_anticoagulation_af_indication() -> None:
    csv = CSVCorRule()
    a = Analysis(label="", claim_text="Anticoagulation is recommended in patients with HF and AF with CHA2DS2-VASc >=2.", domain="cardiology", claim_type="recommendation")  # fmt: skip
    _, r = csv.action(a, {})
    assert a.validation_status in ("passed", "blocked"), "Should have matched CSV AF entry"
    assert r.action_taken


def test_implicit_cor_map_structure() -> None:
    from heartsafe_rag.validation.rules import _IMPLICIT_COR
    assert len(_IMPLICIT_COR) == 6, "Should have 6 COR mappings"
    assert any(cor == "Class 3: No Benefit" for _, cor in _IMPLICIT_COR), "Should map 'no benefit' to Class 3: No Benefit"
    assert any(cor == "Class 3: Harm" for _, cor in _IMPLICIT_COR), "Should map 'contraindicated' to Class 3: Harm"


def test_csv_esa_matches_correctly_not_iron_or_mra() -> None:
    """CSV rule should match new 'Erythropoietin-stimulating agents' entry, not 'Iron IV' or 'MRA'."""
    csv = CSVCorRule()
    a = Analysis(
        label="",
        claim_text=(
            "Within that section, it specifically states that in patients with heart failure "
            "and anemia, erythropoietin-stimulating agents should not be used to improve "
            "morbidity and mortality."
        ),
        domain="cardiology",
        claim_type="recommendation",
    )
    _, r = csv.action(a, {})
    assert r.action_taken, "CSV rule should fire on ESA claim via the ESA CSV entry"
    assert r.message and "Erythropoietin-stimulating agents" in r.message, (
        "Should match the ESA entry, not Iron IV or MRA"
    )


def test_answer_cor_check_catches_invalid_letter_format() -> None:
    """Cross-claim rule should block 'Class B' as non-standard COR format."""
    from heartsafe_rag.validation.rules import AnswerCORCrossCheckRule
    from vlm_guard import Analysis
    rule = AnswerCORCrossCheckRule()
    answer = "Erythropoietin-stimulating agents should not be used (Class B, Level R)."
    claims = [Analysis(label="", claim_text="ESAs should not be used.", domain="cardiology", claim_type="recommendation")]
    _, _, result = rule.action(claims, answer, {})
    assert result.action_taken, "Should detect invalid 'Class B' format"
    assert result.action_type == "block", "Should block non-standard COR"


def test_answer_cor_check_passes_valid_cor() -> None:
    """Cross-claim rule should pass valid 'Class 3: No Benefit' format."""
    from heartsafe_rag.validation.rules import AnswerCORCrossCheckRule
    from vlm_guard import Analysis
    rule = AnswerCORCrossCheckRule()
    answer = "Anticoagulation is not recommended (Class 3: No Benefit, Level B-R) for HF patients in sinus rhythm."
    claims = [Analysis(label="", claim_text="Anticoagulation not recommended.", domain="cardiology", claim_type="recommendation")]
    _, _, result = rule.action(claims, answer, {})
    assert result.action_taken, "Should match anticoagulation CSV entry"
    assert result.action_type == "pass", "Should pass correct COR"


def test_answer_cor_check_blocks_wrong_cor() -> None:
    """Cross-claim rule should block when COR mismatches CSV."""
    from heartsafe_rag.validation.rules import AnswerCORCrossCheckRule
    from vlm_guard import Analysis
    rule = AnswerCORCrossCheckRule()
    answer = "Beta blockers are recommended (Class 2b, Level B) for HFmrEF patients."
    claims = [Analysis(label="", claim_text="Beta blockers for HFmrEF.", domain="cardiology", claim_type="recommendation")]
    _, _, result = rule.action(claims, answer, {})
    assert result.action_taken, "Should fire on beta blocker HFmrEF"
    assert result.action_type == "block", "Should block - 'may be considered' (2b) is wrong for HFmrEF (Class 2a)"


def test_answer_cor_check_esa_generated_example() -> None:
    from heartsafe_rag.validation.rules import AnswerCORCrossCheckRule
    from vlm_guard import Analysis
    rule = AnswerCORCrossCheckRule()
    answer = "The guideline recommends against using erythropoietin-stimulating agents in patients with heart failure and anemia to improve morbidity and mortality (Class B, Level R)."
    claims = [Analysis(label="", claim_text="ESAs should not be used.", domain="cardiology", claim_type="recommendation")]
    _, _, result = rule.action(claims, answer, {})
    assert result.action_taken, "Should detect ESA in answer"
    assert result.action_type == "block", "Should block invalid 'Class B' format"


def test_answer_cor_check_blocks_class3_harm_as_no_benefit() -> None:
    from heartsafe_rag.validation.rules import AnswerCORCrossCheckRule, _cor_matches
    assert not _cor_matches("class 3: harm", "class 3: no benefit"), \
        "class 3: harm should not match class 3: no benefit"
    from vlm_guard import Analysis
    rule = AnswerCORCrossCheckRule()
    answer = "Erythropoietin-stimulating agents are not recommended (Class 3 Harm, Level B-R)."
    claims = [Analysis(label="", claim_text="ESAs are not recommended.", domain="cardiology", claim_type="recommendation")]
    _, _, result = rule.action(claims, answer, {})
    assert result.action_taken, "Should detect ESA in answer"
    assert result.action_type == "block", \
        "Should block 'Class 3 Harm' when CSV says Class 3: No Benefit"


def test_csv_iron_deficiency_claim_matches_correctly() -> None:
    """CSV rule should match 'Iron IV' when claim explicitly mentions iron deficiency."""
    csv = CSVCorRule()
    a = Analysis(
        label="",
        claim_text=(
            "IV iron is recommended for patients with HFrEF and iron deficiency "
            "defined as ferritin <100 ng/mL or TSAT <20%."
        ),
        domain="cardiology",
        claim_type="recommendation",
    )
    _, r = csv.action(a, {})
    assert r.action_taken, "Should match 'Iron IV' CSV entry when 'iron' appears as word boundary"
