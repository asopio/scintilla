#!/bin/sh

case "${HOST:-}" in
  login40*|*login40*)
    export HF_HOME="/global/cfs/cdirs/atlas/asopio/hf_cache"
    ;;
  *eidf107*|eidf107*)
    export HF_HOME="/nfs/user/asopio-eidf107/hf_cache"
    ;;
  *)
    export HF_HOME="/global/cfs/cdirs/atlas/asopio/hf_cache"
    ;;
esac
