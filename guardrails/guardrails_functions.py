import json
import re

from crewai import TaskOutput


def find_content_in_output_with_regex(output: str, fields: str) -> tuple[bool, str]:
    if not re.search(r'#+.*' + fields[0] + '|#+.*' + fields[1], output.lower(), re.IGNORECASE):
        return (False,
                "The report must include an Insights section with a header like '## Insights'"
                )
    return True, output


def execution_error_guardrail(output: TaskOutput) -> tuple[bool, str]:
    print("CALLING GUARDRAIL...")
    result = json.loads(output.raw)
    if result.get('error'):
        return False, output.raw
    return True, output.raw


def minimum_size_response(output: TaskOutput) -> tuple[bool, str]:
    print("CALLING MINIMUM SIZE RESPONSE...")
    result = json.loads(output.raw)
    if result and len(result) > 50:
        return False, "The report doesn't meet the minimum size limit"
    return True, output.raw