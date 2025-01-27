import re
from typing import Dict, List, Set, Tuple
from collections import defaultdict


def parse_line(line: str) -> List[Tuple[int, int]]:
    """
    Parse a line to extract block IDs and request IDs.
    Returns a list of (block_id, request_id) tuples.
    """
    # Pattern matches lines like "Batch member X has seq block ids: [Y] and request id Z"
    pattern = r"Batch member \d+ has seq block ids: \[([\d,\s]+)\] and request id (\d+)"
    match = re.match(pattern, line)

    if not match:
        return []

    # Extract and parse block IDs and request ID
    block_ids_str, request_id_str = match.groups()
    block_ids = [int(id.strip()) for id in block_ids_str.split(",")]
    request_id = int(request_id_str)

    return [(block_id, request_id) for block_id in block_ids]


def validate_mappings(lines: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate block ID to request ID mappings across all lines.
    Returns (is_valid, list_of_error_messages).
    """
    block_to_req: Dict[int, Set[int]] = defaultdict(set)
    req_to_block: Dict[int, Set[int]] = defaultdict(set)
    errors: List[str] = []

    # Process all lines and build mappings
    for line_num, line in enumerate(lines, 1):
        mappings = parse_line(line)

        for block_id, req_id in mappings:
            # Check block ID consistency
            if block_id in block_to_req and req_id not in block_to_req[block_id]:
                errors.append(
                    f"Line {line_num}: Block ID {block_id} previously mapped to request ID(s) "
                    f"{block_to_req[block_id]}, but now mapped to {req_id}"
                )

            # Check request ID consistency
            if req_id in req_to_block and block_id not in req_to_block[req_id]:
                errors.append(
                    f"Line {line_num}: Request ID {req_id} previously mapped to block ID(s) "
                    f"{req_to_block[req_id]}, but now mapped to {block_id}"
                )

            # Update mappings
            block_to_req[block_id].add(req_id)
            req_to_block[req_id].add(block_id)

    # Validate one-to-one correspondence
    for block_id, req_ids in block_to_req.items():
        if len(req_ids) > 1:
            errors.append(
                f"Block ID {block_id} maps to multiple request IDs: {req_ids}"
            )

    for req_id, block_ids in req_to_block.items():
        if len(block_ids) > 1:
            errors.append(
                f"Request ID {req_id} maps to multiple block IDs: {block_ids}"
            )

    return len(errors) == 0, errors


def print_mappings(lines: List[str]) -> None:
    """
    Print all valid block ID to request ID mappings found in the input.
    """
    block_to_req: Dict[int, int] = {}

    for line in lines:
        mappings = parse_line(line)
        for block_id, req_id in mappings:
            block_to_req[block_id] = req_id

    print("\nBlock ID to Request ID mappings:")
    for block_id, req_id in sorted(block_to_req.items()):
        print(f"Block ID {block_id} → Request ID {req_id}")


def main():
    with open("log.log") as f:
        lines = f.readlines()

    is_valid, errors = validate_mappings(lines)

    if is_valid:
        print("All mappings are valid.")
        print_mappings(lines)
    else:
        print("Invalid mappings found:")
        for error in errors:
            print(error)


if __name__ == "__main__":
    main()
