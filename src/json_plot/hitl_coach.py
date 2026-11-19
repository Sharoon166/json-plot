"""Real-time review guidance from the annotated JSON.

The human checks the model's boxes. The JSON says which marks, schedule rows,
and callouts belong on a page. These notes turn that into the next action.
"""

from __future__ import annotations

from dataclasses import dataclass

from json_plot.relationships import RelationshipGraph, callout_targets


@dataclass(frozen=True)
class CoachNote:
    action: str
    title: str
    body: str


def page_brief(
    graph: RelationshipGraph,
    page: int,
    pending: int = 0,
    proposals: int = 0,
) -> CoachNote:
    if not graph.nodes:
        return CoachNote(
            "wait",
            "Load the JSON answer key",
            "Open a project with the schedule, output, and reference JSON. Those files say which marks and callouts belong on each page. You then accept, reject, or draw what the model got wrong.",
        )
    marks = _labels_for_page(graph, page, "on_plan")
    schedule = _labels_for_page(graph, page, "on_schedule")
    missing = _labels_for_page(graph, page, "missing")
    bad_calls = _labels_for_page(graph, page, "bad_call")
    calls = _labels_for_page(graph, page, "calls")
    lines = [
        "Accept a box that sits on a real tag, door, schedule row, callout, or reference. Reject a box that is empty or the wrong thing. If the JSON names something on this page and there is no box, draw it with Mark missed object.",
    ]
    if marks:
        lines.append(f"{len(marks)} marks belong on this page, including {_sample(marks)}.")
    if schedule:
        lines.append(f"{len(schedule)} schedule rows belong here, including {_sample(schedule)}.")
    if missing:
        lines.append(f"{len(missing)} are missing from the schedule: {_sample(missing)}. Accept them when you can see them.")
    if bad_calls:
        lines.append(f"Bad callouts to confirm: {_sample(bad_calls)}.")
    elif calls:
        lines.append(f"Sheet callouts on this page: {_sample(calls)}.")
    if proposals == 0:
        lines.append("Run page to create proposals, then decide them one by one.")
    elif pending:
        lines.append(f"{pending} of {proposals} proposals still need your decision.")
    else:
        lines.append(f"All {proposals} proposals on this page have a decision.")
    title = f"Page {page}: check each box against the JSON"
    return CoachNote("check" if pending or proposals == 0 else "done", title, " ".join(lines))


def detection_brief(
    graph: RelationshipGraph,
    page: int,
    text: str,
    class_name: str,
    decision: str,
    ocr_confidence: float | None,
) -> CoachNote:
    label = class_name.replace("_", " ")
    shown = text.strip() or "no text"
    unsure = ""
    if ocr_confidence is not None and ocr_confidence < 0.5:
        unsure = "OCR is unsure, so read the drawing before you decide. "

    if not graph.nodes:
        return CoachNote(
            "check",
            f"Look at this {label}",
            f"{unsure}The JSON answer key is not loaded, so compare the box with the drawing yourself. Accept it only if it is a tag, door, schedule row, callout, or reference.",
        )

    match = _best_match(graph, page, text)
    if match is None:
        action = "reject"
        title = f"Reject this {label} unless you can read it"
        body = (
            f"{unsure}Nothing in the JSON uses “{shown}” on page {page}. "
            "Reject the box if it is empty or the wrong object. Accept it only when you can see a real tag, door, schedule row, callout, or reference, and leave a note."
        )
    else:
        action, title, body = _matched_note(match, page, label, shown, unsure)

    if decision == "accepted":
        return CoachNote(
            "done" if action == "accept" else "check",
            f"Accepted. {title}",
            body if action == "accept" else f"You accepted a box the JSON does not fully support. {body}",
        )
    if decision == "rejected":
        follow = " If the object is still on the page, draw it with Mark missed object." if action == "accept" else " Move on to the next pending box."
        return CoachNote("mark" if action == "accept" else "done", f"Rejected. {title}", body + follow)
    return CoachNote(action, title, body)


@dataclass(frozen=True)
class _Match:
    kind: str
    name: str
    on_plan: list[int]
    on_schedule: list[int]
    missing_here: bool
    call_type: str | None


def _matched_note(match: _Match, page: int, label: str, shown: str, unsure: str) -> tuple[str, str, str]:
    if match.kind in {"sheet", "index"}:
        if match.call_type == "bad_call":
            return (
                "check",
                f"Confirm the bad callout {match.name}",
                f"{unsure}The JSON marks “{shown}” as a bad sheet callout for {match.name}. Accept when the box is that callout and set the class to reference. Reject when the box is a different object.",
            )
        if match.call_type == "missing_index":
            return (
                "check",
                f"{match.name} is missing from the index",
                f"{unsure}Accept the box only if it is the {match.name} reference. The JSON says this code is not in the sheet index.",
            )
        return (
            "accept",
            f"Accept if this box is the {match.name} callout",
            f"{unsure}The JSON links page {page} to {match.name}. Accept a box that sits on that callout. Reject it if the box missed the symbol.",
        )

    here_plan = page in match.on_plan
    here_schedule = page in match.on_schedule
    schedule_pages = ", ".join(f"page {number}" for number in match.on_schedule[:3]) or "no schedule page"
    plan_pages = ", ".join(f"page {number}" for number in match.on_plan[:3]) or "no plan page"
    if match.missing_here:
        return (
            "accept",
            f"Accept {match.name} if you can see it",
            f"{unsure}The JSON says {match.name} is on this page and missing from the schedule. Accept the box when it sits on that mark. That teaches the model a real tag the schedule does not list.",
        )
    if here_plan and label == "schedule entry":
        return (
            "check",
            f"{match.name} is a plan tag, not a schedule row",
            f"{unsure}It is drawn on {plan_pages} and listed on {schedule_pages}. Change the class to tag if the box is on the tag. Reject the box if it is somewhere else.",
        )
    if here_schedule and label in {"tag", "door reference"}:
        return (
            "check",
            f"{match.name} is a schedule row on this page",
            f"{unsure}The schedule entry is here and the tag is drawn on {plan_pages}. Set the class to schedule entry if the box is on the table. Reject it if the box missed the row.",
        )
    if here_plan or here_schedule:
        where = "on this plan" if here_plan else "in the schedule on this page"
        other = schedule_pages if here_plan else plan_pages
        return (
            "accept",
            f"Accept if the box is on {match.name}",
            f"{unsure}{match.name} belongs {where}. The other end of the link is {other}. Accept a box that covers it. Reject a box that only sits nearby, then draw the right one if the model missed it.",
        )
    return (
        "reject",
        f"{match.name} belongs on another page",
        f"{unsure}The JSON places it on {plan_pages} and lists it on {schedule_pages}, not on page {page}. Reject this box unless the drawing clearly shows it here too.",
    )


def _best_match(graph: RelationshipGraph, page: int, text: str) -> _Match | None:
    for token in _candidates(text):
        mark = graph.nodes.get(f"mark:{token}")
        if mark is not None:
            return _mark_match(graph, page, mark.label)
        for kind in ("sheet", "index"):
            node = graph.nodes.get(f"{kind}:{token}")
            if node is not None:
                return _code_match(graph, page, kind, node.label)
    return None


def _mark_match(graph: RelationshipGraph, page: int, name: str) -> _Match:
    node_id = f"mark:{name}"
    on_plan, on_schedule, missing_here = [], [], False
    for edge in graph.edges.values():
        if edge.src != node_id or not edge.dst.startswith("page:"):
            continue
        number = int(edge.dst.split(":", 1)[1])
        if edge.type == "on_plan":
            on_plan.append(number)
        elif edge.type == "on_schedule":
            on_schedule.append(number)
        elif edge.type == "missing" and number == page:
            missing_here = True
    return _Match("mark", name, sorted(set(on_plan)), sorted(set(on_schedule)), missing_here, None)


def _code_match(graph: RelationshipGraph, page: int, kind: str, name: str) -> _Match:
    node_id = f"{kind}:{name}"
    page_id = f"page:{page}"
    call_type = None
    priority = {"bad_call": 4, "missing_index": 3, "unresolved": 2, "calls": 1, "indexes": 1}
    for edge in graph.edges.values():
        if {edge.src, edge.dst} != {page_id, node_id}:
            continue
        if priority.get(edge.type, 0) > priority.get(call_type or "", 0):
            call_type = edge.type
    return _Match(kind, name, [], [], False, call_type)


def _labels_for_page(graph: RelationshipGraph, page: int, edge_type: str) -> list[str]:
    page_id = f"page:{page}"
    labels = []
    for edge in graph.edges.values():
        if edge.type != edge_type or page_id not in (edge.src, edge.dst):
            continue
        other_id = edge.dst if edge.src == page_id else edge.src
        node = graph.nodes.get(other_id)
        if node is not None and node.type != "page":
            labels.append(node.label)
    return list(dict.fromkeys(labels))


def _candidates(text: str) -> list[str]:
    raw = text.strip()
    if not raw:
        return []
    found = [raw]
    found.extend(code for _, code, _ in callout_targets(raw))
    cleaned = "".join(character if character.isalnum() or character in ".- " else " " for character in raw)
    found.extend(raw.split())
    found.extend(cleaned.split())
    return list(dict.fromkeys(token for token in found if token))


def _sample(labels: list[str]) -> str:
    shown = labels[:5]
    extra = len(labels) - len(shown)
    text = ", ".join(shown)
    if extra:
        text += f", and {extra} more"
    return text
