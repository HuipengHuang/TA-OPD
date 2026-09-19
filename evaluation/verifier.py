from math_verify import parse, verify
# =========================
# Extract answer region
# =========================
def find_first_boxed_end(text: str) -> int | None:
    """
    Find the end position of the first \\boxed{...}, supporting nested braces.
    Return the end index, or None if not found.
    """
    start = text.find(r"\boxed")
    if start == -1:
        return None

    brace_start = text.find("{", start)
    if brace_start == -1:
        return None

    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1

    return None


def extract_answer_region(text: str) -> str:
    """
    Logic:
    1. If any final/answer marker exists:
       - Extract from the first marker to the second marker.
       - If there is no second marker, keep from the first marker
         to the end of the first \\boxed{...}.
    2. Otherwise return the original text.
    """
    if text is None:
        return ""

    final_markers = [
        "**Final Answer**",
        "**Final Answer:**",
        "**Answer**",
        "**Answer:**",
    ]

    # find the earliest occurring marker
    first_candidates = [
        (text.find(marker), marker)
        for marker in final_markers
        if text.find(marker) != -1
    ]

    if first_candidates:
        first_final, first_marker = min(first_candidates, key=lambda x: x[0])

        # find the next occurrence of any marker after the first marker
        second_candidates = []
        search_start = first_final + len(first_marker)

        for marker in final_markers:
            pos = text.find(marker, search_start)
            if pos != -1:
                second_candidates.append(pos)

        if second_candidates:
            second_final = min(second_candidates)
            return text[first_final:second_final].strip()

        tail = text[first_final:]
        boxed_end = find_first_boxed_end(tail)

        if boxed_end is not None:
            return tail[:boxed_end].strip()

        # fallback: avoid parsing extremely long repetitive text
        return tail.strip()

    return text.strip()


# =========================
# Verification
# =========================

def extract_first_boxed_content(text: str) -> str | None:
    """
    Extract the content inside the first \\boxed{...}.
    Supports nested braces, e.g. \\boxed{\\frac{1}{2}}.
    """
    if text is None:
        return None

    start = text.find(r"\boxed")
    if start == -1:
        return None

    brace_start = text.find("{", start)
    if brace_start == -1:
        return None

    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1:i].strip()

    return None


def safe_parse(text: str, parsing_timeout: int = 5):
    try:
        return parse(
            text,
            parsing_timeout=parsing_timeout,
        )
    except Exception:
        boxed_content = extract_first_boxed_content(text)

        if boxed_content is None:
            return []

        try:
            return parse(
                boxed_content,
                parsing_timeout=parsing_timeout,
            )
        except Exception:
            return []


from concurrent.futures import ProcessPoolExecutor
from functools import partial


def judge_one_response(response: str, golden_parsed, parsing_timeout: int = 30) -> bool:
    """
    First try to parse the original full response directly.
    If it fails or verification fails, fall back to extracted-region parsing.
    """
    if response is None:
        response = ""

    # 1. First try direct parse on the full response
    try:
        pred_parsed = safe_parse(response, parsing_timeout=parsing_timeout)
        if verify(golden_parsed, pred_parsed):
            return True
    except Exception:
        pass

    # 2. Fallback: extract answer region, then parse
    try:
        processed_response = extract_answer_region(response)
        pred_parsed = safe_parse(processed_response, parsing_timeout=parsing_timeout)
        return bool(verify(golden_parsed, pred_parsed))
    except Exception:
        return False


def _judge_one_response_worker(args):
    response, golden_parsed, parsing_timeout = args
    return judge_one_response(
        response=response,
        golden_parsed=golden_parsed,
        parsing_timeout=parsing_timeout,
    )


def labeling_responses(
    responses: list[str],
    golden_answer: str,
    parsing_timeout: int = 30,
    num_workers: int = 16,
) -> list[bool]:
    golden_parsed = safe_parse(
        "$" + str(golden_answer) + "$",
        parsing_timeout=parsing_timeout,
    )

    if len(responses) == 0:
        return []

    if num_workers <= 1 or len(responses) <= 1:
        return [
            judge_one_response(
                response=response,
                golden_parsed=golden_parsed,
                parsing_timeout=parsing_timeout,
            )
            for response in responses
        ]

    tasks = [
        (response, golden_parsed, parsing_timeout)
        for response in responses
    ]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        correctness_list = list(executor.map(_judge_one_response_worker, tasks))

    return correctness_list