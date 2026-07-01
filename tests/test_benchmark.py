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
