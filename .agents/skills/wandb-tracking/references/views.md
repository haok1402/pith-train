# wandb views via GraphQL

Views are managed through GraphQL: `api._service_api.execute_graphql(query_str, variables)` takes a query string and a variables dict and returns the parsed `data` field.

## nw slug <-> view name

- Workspace URL `https://wandb.ai/<entity>/<project>?nw=<slug>`.
- Internal view name is `nw-<slug>-v`; the human label is `displayName`.
- The default workspace view ends in `-w` (e.g. `nw-<user>-w`) rather than `-v`. It is a project-view too, so it appears in the list and works as a clone template.

## Listing views

```graphql
query Views($entityName: String!, $projectName: String!) {
  project(name: $projectName, entityName: $entityName) {
    allViews(viewType: "project-view") {
      edges { node { id name displayName spec } }
    }
  }
}
```

`spec` is a JSON string. The runset lives at `spec.section.runSets[0]`.

## What a view shows (selection semantics)

- `runSets[0].filters` empty => every run in the project is in the table.
- `runSets[0].selections = {root, bounds, tree}` narrows what is plotted.
- With `root == 1`, `tree` is the list of **hidden** (eye-closed) run ids, so:

  **shown = all_run_ids - tree**

To narrow a view to a set of runs, set `tree = sorted(all_ids - show_ids)`.

## Creating / updating (upsertView)

```graphql
mutation Up($input: UpsertViewInput!) {
  upsertView(input: $input) { view { id name displayName } inserted }
}
```

`UpsertViewInput` fields that matter:

- `entityName`, `projectName`, `type: "project-view"` (where it lives).
- `name`: `nw-<slug>-v`. Mint a fresh 11-char slug for a new view.
- `displayName`: the human label.
- `spec`: JSON string (clone an existing view's spec, then rewrite the tree).
- `createdUsing`: `WANDB_SDK`.
- `id`: pass it to update in place (keeps the same URL); omit to create.
  `inserted` in the response is `true` for create, `false` for update.

Cloning an existing view's spec carries the project's panel layout, so a new view opens on the same charts.

## Deleting

```graphql
mutation Del($input: DeleteViewInput!) { deleteView(input: $input) { success } }
```

Input is `{ id }`. Views are cheap and reversible, but creating/updating one is a write to the user's project, so confirm the runs and name first.
