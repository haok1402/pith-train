"""
Create / update / list / delete wandb project views (the ?nw=<slug> saved
workspaces).

    python scripts/views.py list   <entity/project>
    python scripts/views.py show   <entity/project> <nw_slug>
    python scripts/views.py set    <entity/project> --show RUN[,RUN...] --name DISPLAYNAME
                                    [--from-view NW_SLUG] [--update NW_SLUG]
    python scripts/views.py delete <entity/project> <nw_slug>

Key facts (see references/views.md for the full anatomy):
  - URL ?nw=<slug> maps to internal view name nw-<slug>-v.
  - A runset with empty filters means ALL runs are in the table; the selections
    object narrows what's shown. With selections.root == 1, the tree lists the
    HIDDEN (eye-closed) runs, so shown = all_runs - tree.
  - set clones a template spec (an existing view, --from-view, or a built-in
    fallback), rewrites selections.tree = all_ids - show_ids, and upserts.
    Pass --update <nw_slug> to modify an existing view in place (same URL).
"""

import argparse
import json
import random
import string
import sys

import wandb

VIEWS_Q = """
query Views($entityName: String!, $projectName: String!) {
  project(name: $projectName, entityName: $entityName) {
    allViews(viewType: "project-view") {
      edges { node { id name displayName spec } }
    }
  }
}
"""

UPSERT_M = """
mutation Up($input: UpsertViewInput!) {
  upsertView(input: $input) { view { id name displayName } inserted }
}
"""

DELETE_M = """
mutation Del($input: DeleteViewInput!) { deleteView(input: $input) { success } }
"""

# Fallback spec (auto panels), used when the project has no view to clone from.
# tree and id are filled in by the set command.
FALLBACK_SPEC = {
    "section": {
        "name": "",
        "version": 1,
        "openRunSet": 0,
        "openViz": True,
        "runSets": [
            {
                "id": "REPLACE_ID",
                "runFeed": {
                    "version": 2,
                    "columnVisible": {"run:name": False},
                    "columnPinned": {},
                    "columnWidths": {},
                    "columnOrder": [],
                    "pageSize": 20,
                    "onlyShowSelected": False,
                    "metricValences": {},
                },
                "search": {"query": ""},
                "searchHistory": [],
                "name": "Run set",
                "enabled": True,
                "runNameTruncationType": "Middle",
                "pinnedRunIds": [],
                "filters": {"filterFormat": "filterV2", "filters": []},
                "grouping": [],
                "sort": {
                    "keys": [{"key": {"section": "run", "name": "createdAt"}, "ascending": False}]
                },
                "selections": {"root": 1, "bounds": [], "tree": []},
                "expandedRowAddresses": [],
            }
        ],
        "panelBankConfig": {
            "state": 1,
            "settings": {
                "showEmptySections": False,
                "sortAlphabetically": False,
                "defaultMoveToSectionName": "train",
            },
            "sections": [],
        },
        "panelBankSectionConfig": {
            "__id__": "rps",
            "name": "Report Panels",
            "isOpen": False,
            "sorted": 0,
            "pinned": False,
            "panels": [],
        },
        "customRunColors": {},
        "customRunNames": {},
        "workspaceSettings": {"linePlot": {}},
    }
}


def slug(n):
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=n))


def fetch_views(api, entity, project):
    res = api._service_api.execute_graphql(VIEWS_Q, {"entityName": entity, "projectName": project})
    return [edge["node"] for edge in res["project"]["allViews"]["edges"]]


def shown_hidden(spec, id2name):
    tree = set(spec["section"]["runSets"][0]["selections"]["tree"])
    shown = sorted(id2name.get(i, i) for i in set(id2name) - tree)
    hidden = sorted(id2name.get(i, i) for i in tree)
    return shown, hidden


def main():
    ap = argparse.ArgumentParser(usage=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").add_argument("path")
    for c in ("show", "delete"):
        sp = sub.add_parser(c)
        sp.add_argument("path")
        sp.add_argument("nw")
    sp = sub.add_parser("set")
    sp.add_argument("path")
    sp.add_argument("--show", required=True, help="run ids/names to make visible")
    sp.add_argument("--name", help="display name (required for new views)")
    sp.add_argument("--from-view", help="nw slug to clone panels/layout from")
    sp.add_argument("--update", help="nw slug to modify in place (keeps the URL)")
    args = ap.parse_args()

    entity, project = args.path.split("/", 1)
    api = wandb.Api()
    id2name = {r.id: r.name for r in api.runs(args.path)}

    if args.cmd == "list":
        for n in fetch_views(api, entity, project):
            nw = (
                n["name"][3:-2] if n["name"].startswith("nw-") and n["name"].endswith("-v") else "?"
            )
            print(f"- {n['displayName']!r}  (nw={nw}, name={n['name']})")
            try:
                shown, hidden = shown_hidden(json.loads(n["spec"]), id2name)
                print(f"    shown : {shown}")
                print(f"    hidden: {hidden}")
            except (KeyError, json.JSONDecodeError):
                print("    (no runset selection in spec)")
        return

    views = fetch_views(api, entity, project)

    if args.cmd in ("show", "delete"):
        node = next((v for v in views if v["name"] == f"nw-{args.nw}-v"), None)
        if node is None:
            raise SystemExit(f"view nw={args.nw} not found in {args.path}")
        if args.cmd == "show":
            shown, hidden = shown_hidden(json.loads(node["spec"]), id2name)
            print(f"{node['displayName']!r}  (nw={args.nw})")
            print(f"  shown : {shown}")
            print(f"  hidden: {hidden}")
        else:
            res = api._service_api.execute_graphql(DELETE_M, {"input": {"id": node["id"]}})
            print(
                f"deleted {node['displayName']!r} (nw={args.nw}): success={res['deleteView']['success']}"
            )
        return

    # set (create or update)
    if not args.update and not args.name:
        sys.exit("--name is required when creating a new view")
    name2id = {v: k for k, v in id2name.items()}
    show_ids = {sel if sel in id2name else name2id[sel] for sel in args.show.split(",")}
    tree = sorted(set(id2name) - show_ids)

    template = None
    if args.update:
        template = next((v for v in views if v["name"] == f"nw-{args.update}-v"), None)
        if template is None:
            raise SystemExit(f"--update nw={args.update} not found")
    elif args.from_view:
        template = next((v for v in views if v["name"] == f"nw-{args.from_view}-v"), None)
        if template is None:
            raise SystemExit(f"--from-view nw={args.from_view} not found")
    elif views:
        template = views[0]
    spec = json.loads(template["spec"]) if template else json.loads(json.dumps(FALLBACK_SPEC))
    spec["section"]["runSets"][0]["selections"] = {"root": 1, "bounds": [], "tree": tree}

    if args.update:
        nw = args.update
        inp = {
            "id": template["id"],
            "entityName": entity,
            "projectName": project,
            "type": "project-view",
            "name": template["name"],
            "displayName": args.name or template["displayName"],
            "spec": json.dumps(spec),
        }
    else:
        nw = slug(11)
        spec["section"]["runSets"][0]["id"] = slug(9)
        inp = {
            "entityName": entity,
            "projectName": project,
            "type": "project-view",
            "name": f"nw-{nw}-v",
            "displayName": args.name,
            "createdUsing": "WANDB_SDK",
            "spec": json.dumps(spec),
        }

    res = api._service_api.execute_graphql(UPSERT_M, {"input": inp})
    v = res["upsertView"]["view"]
    print(
        f"{'inserted' if res['upsertView']['inserted'] else 'updated'}: {v['displayName']!r}  (nw={nw})"
    )
    print(f"URL: https://wandb.ai/{args.path}?nw={nw}")
    print(f"shown : {sorted(id2name[i] for i in show_ids)}")


if __name__ == "__main__":
    main()
