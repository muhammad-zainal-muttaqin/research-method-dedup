"""Build Brand-New-Dataset-YOLO/ from cleaned 05 Mei 2026/ sources.

Layout:
  Brand-New-Dataset-YOLO/
    data.yaml
    images/{train,val,test}/    <- 05 Mei 2026/images/...
    labels/{train,val,test}/    <- 05 Mei 2026/Output TXT/...
    json/                       <- 05 Mei 2026/Output JSON/ (953 files)

Method: copy. Source untouched. Fails fast if target already exists.
"""
import json, os, shutil, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "05 Mei 2026")
DST = os.path.join(ROOT, "Brand-New-Dataset-YOLO")

DATA_YAML = """path: .
train: images/train
val: images/val
test: images/test

nc: 4
names:
  0: B1
  1: B2
  2: B3
  3: B4
"""


def main():
    if os.path.exists(DST):
        sys.exit(f"[abort] {DST} already exists. Remove or rename first.")

    os.makedirs(DST)
    print(f"[mkdir] {DST}")

    # images
    os.makedirs(os.path.join(DST, "images"))
    for s in ("train", "val", "test"):
        src = os.path.join(SRC, "images", s)
        dst = os.path.join(DST, "images", s)
        shutil.copytree(src, dst)
        print(f"  [copy] images/{s}: {len(os.listdir(dst))} files")

    # labels (from Output TXT)
    os.makedirs(os.path.join(DST, "labels"))
    for s in ("train", "val", "test"):
        src = os.path.join(SRC, "Output TXT", s)
        dst = os.path.join(DST, "labels", s)
        shutil.copytree(src, dst)
        print(f"  [copy] labels/{s}: {len(os.listdir(dst))} files")

    # json (only .json files, drop any non-json artifacts)
    src_json = os.path.join(SRC, "Output JSON")
    dst_json = os.path.join(DST, "json")
    os.makedirs(dst_json)
    n = 0
    for f in os.listdir(src_json):
        if f.endswith(".json"):
            shutil.copy2(os.path.join(src_json, f), os.path.join(dst_json, f))
            n += 1
    print(f"  [copy] json: {n} files")

    # data.yaml
    with open(os.path.join(DST, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(DATA_YAML)
    print(f"  [write] data.yaml")

    # verify
    print("\n=== Verification ===")
    expect = {"train": 2780, "val": 620, "test": 592}
    ok = True
    for s, want in expect.items():
        ni = len(os.listdir(os.path.join(DST, "images", s)))
        nl = len(os.listdir(os.path.join(DST, "labels", s)))
        si = {f.rsplit('.',1)[0] for f in os.listdir(os.path.join(DST,"images",s))}
        sl = {f.rsplit('.',1)[0] for f in os.listdir(os.path.join(DST,"labels",s))}
        diff = len(si ^ sl)
        status = "OK" if (ni == want and nl == want and diff == 0) else "FAIL"
        if status == "FAIL": ok = False
        print(f"  {s}: img={ni} lbl={nl} stem_diff={diff} [{status}]")
    nj = len(os.listdir(dst_json))
    print(f"  json: {nj} [{'OK' if nj == 953 else 'FAIL'}]")

    # spot-check anno vs label
    import random
    random.seed(0)
    sample = random.sample([f for f in os.listdir(dst_json) if f.endswith('.json')], 10)
    mism = 0
    for jf in sample:
        d = json.load(open(os.path.join(dst_json, jf), encoding='utf-8'))
        for side, sd in d['images'].items():
            n_anno = len(sd.get('annotations', []))
            lbl = sd.get('label_file')
            for s in ('train','val','test'):
                p = os.path.join(DST, "labels", s, lbl)
                if os.path.exists(p):
                    nl = sum(1 for l in open(p).read().splitlines() if l.strip())
                    if nl != n_anno: mism += 1
                    break
    print(f"  spot-check 10 trees: anno↔label mismatch = {mism} [{'OK' if mism==0 else 'FAIL'}]")
    print(f"\n{'ALL OK' if ok and mism == 0 else 'ISSUES FOUND'}")


if __name__ == "__main__":
    main()
