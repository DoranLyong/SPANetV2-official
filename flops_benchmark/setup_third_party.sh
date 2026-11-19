#!/bin/bash
# Clone the comparison repositories at the commits used for the measurement and apply the
# one import-path fix (GFNet, numpy>=2). SPANetV2 itself is the repository this folder lives in.
#   bash setup_third_party.sh
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p third_party && cd third_party

clone() {  # url dir commit
    [ -d "$2" ] || git clone -q "$1" "$2"
    git -C "$2" checkout -q "$3"
    echo "$2 $(git -C "$2" rev-parse HEAD)"
}
clone https://github.com/DoranLyong/SPANet-official.git SPANet-official 886ad24d4d69495401e93e90d2b0a52c6cb718b7
clone https://github.com/sail-sg/metaformer.git          metaformer      64c534ab8f65105fd21713b38e3393a449ca742d
clone https://github.com/okojoalg/dfformer.git           dfformer        2b757c511a1bf1e66a12d494f3b0c42daa5b3636
clone https://github.com/raoyongming/GFNet.git           GFNet           2de48b626ea805de82a7f389fed930af1ae0340d

# GFNet/gfnet.py: `from numpy.lib.arraypad import pad` fails on numpy>=2 (the name `pad` is never used).
git -C GFNet apply --check ../../patches/GFNet_numpy_import.patch 2>/dev/null \
    && git -C GFNet apply ../../patches/GFNet_numpy_import.patch && echo "patched GFNet/gfnet.py" \
    || echo "GFNet patch already applied"
