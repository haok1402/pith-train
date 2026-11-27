#!/bin/bash
# Source from the Slurm worker after creating a job-local destination.
stage_runtime() {
    local packed=$1 stage=$2
    local archive base_python site source_site pid status=0
    local -a pids=()
    read -r base_python < "$packed/python"
    read -r site < "$packed/site"
    read -r source_site < "$packed/source-site"
    test -s "$packed/sha256"
    test -x "$base_python"
    for archive in "$packed"/venv-*.tar.zst; do
        (zstd -dcq < "$archive" | tar -xf - -C "$stage") &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    (( status == 0 )) || return "$status"
    mkdir "$stage/bin"
    ln -s "$base_python" "$stage/bin/python"
    printf '%s\n' "$source_site" > "$stage/$site/zz_shared_site_packages.pth"
    if [[ ! -e $stage/$site/torch/include ]]; then
        ln -s "$source_site/torch/include" "$stage/$site/torch/include"
    fi
}
