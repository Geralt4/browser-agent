from benchmarks.run import _check_expect


def test_expect_match_case_insensitive():
    assert _check_expect("The H1 says: Example Domain", "Example Domain") is True


def test_expect_match_lowercase_result():
    assert _check_expect("the heading is herman melville", "Herman Melville") is True


def test_expect_no_match_different_content():
    assert _check_expect("The page is about something else", "Example Domain") is False


def test_expect_empty_string_treated_as_no_ground_truth():
    assert _check_expect("anything", "") is False


def test_expect_none_treated_as_no_ground_truth():
    assert _check_expect("anything", None) is False


def test_expect_empty_result_does_not_match():
    assert _check_expect("", "Example Domain") is False


def test_expect_none_result_does_not_match():
    assert _check_expect(str(None), "Example Domain") is False


# ── Punctuation-bounded expect values (C4 regression) ───────────────────
# The previous `r"\b" + re.escape(expect) + r"\b"` pattern failed silently
# when `expect` started or ended with a non-word character — there was no
# word char on the `expect` side of the boundary for `\b` to anchor to.
# The fix uses lookbehind/lookahead pairs (`(?<![A-Za-z0-9_])` /
# `(?![A-Za-z0-9_])`) which handle every case uniformly. Each of the
# values below contains a leading or trailing non-word character.


def test_expect_punctuation_paren_surrounded():
    """`expect` is the literal text "(Domain)" — must match when the
    result contains the same parenthetical phrase."""
    assert _check_expect("the Example (Domain) is here", "(Domain)") is True


def test_expect_dollar_amount():
    assert _check_expect("costs $10.00 today", "$10.00") is True


def test_expect_cpp():
    """C++ ends with a non-word char ('+'). The old regex would never
    match because the trailing `\b` has no word char on the `expect` side."""
    assert _check_expect("write C++ code", "C++") is True


def test_expect_hyphenated_model_name():
    """gpt-4o has a hyphen — word boundary still works here because the
    characters adjacent to the hyphen are word chars."""
    assert _check_expect("use gpt-4o for this", "gpt-4o") is True


def test_expect_quoted_phrase():
    assert _check_expect('she said "hello" yesterday', '"hello"') is True


def test_expect_leading_bang():
    assert _check_expect("CSS rule !important applies", "!important") is True


# ── Word-boundary protection (false-positive prevention) ─────────────────
# The lookbehind/lookahead must still prevent substring matches that
# `re.search(r"\b...\b")` would have rejected.


def test_expect_no_substring_match_at_end():
    """`expect='post'` must not match `result='poster'` — the lookbehind
    requires the char BEFORE 'post' to be a non-word char (or start)."""
    assert _check_expect("look at the poster", "post") is False


def test_expect_no_substring_match_in_middle():
    """`expect='pay'` must not match `result='payment'`."""
    assert _check_expect("make a payment", "pay") is False


def test_expect_no_substring_match_submit():
    """`expect='submit'` must not match `result='submitted'`."""
    assert _check_expect("I submitted the form", "submit") is False


def test_expect_no_substring_match_done():
    assert _check_expect("this is undoable", "done") is False
