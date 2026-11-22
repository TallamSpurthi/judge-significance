"""Step 27. Extract the MS MARCO v2 passages that the TREC DL 2023 qrels need, streaming the official corpus.

The corpus (msmarco_v2_passage.tar, 21,768,192,000 bytes) is read once as a stream, from the official URL used
by ir_datasets (msmarco-passage-v2, MD5 05946bac48a8ffee62e160213eab3fda) or from a local copy. Nothing but
the needed passages is written. Each passage id msmarco_passage_BB_OFFSET names its bundle file
msmarco_passage_BB.gz inside the tar and its byte offset in that file uncompressed, so lines are selected by
offset and their pid is then checked against the requested id. The MD5 of the whole stream is checked at the
end; on a mismatch the output is renamed *.INVALID.

--max-bytes stops after that many bytes (a test of the first part of the file; the MD5 is then not checked).

Usage: python scripts/27_extract_passages.py --ids IDS.txt --out PASSAGES.jsonl [--url URL | --tar FILE]
           [--md5 HEX] [--max-bytes N]
IDS.txt: one passage id per line (other lines are ignored).
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import tarfile
import time
import urllib.request

URL = "https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco_v2_passage.tar"
MD5 = "05946bac48a8ffee62e160213eab3fda"
SIZE = 21768192000
_PID = re.compile(r"^msmarco_passage_(\d\d)_(\d+)$")


class HashingReader:
    """File-like wrapper that hashes and counts every byte read, and reports EOF after max_bytes."""

    def __init__(self, raw, max_bytes=None):
        self.raw, self.max_bytes = raw, max_bytes
        self.md5, self.n = hashlib.md5(), 0
        self.t0, self.last = time.time(), 0

    def read(self, size=-1):
        if self.max_bytes is not None:
            left = self.max_bytes - self.n
            if left <= 0:
                return b""
            size = left if size is None or size < 0 else min(size, left)
        chunk = self.raw.read(size)
        self.md5.update(chunk)
        self.n += len(chunk)
        if self.n - self.last >= 1 << 30:
            self.last = self.n
            rate = self.n / max(time.time() - self.t0, 1e-9) / 1e6
            print(f"  read {self.n / 1e9:.1f} GB ({rate:.0f} MB/s)", file=sys.stderr, flush=True)
        return chunk


def needed_offsets(ids):
    """{bundle: {offset: pid}} for valid passage ids; invalid ids are returned separately."""
    want, bad = {}, []
    for pid in ids:
        m = _PID.match(pid)
        if not m:
            bad.append(pid)
            continue
        want.setdefault(m.group(1), {})[int(m.group(2))] = pid
    return want, bad


def extract(stream, want):
    """Yield (pid, record) for every wanted passage found in a tar stream of msmarco_passage_BB.gz members."""
    with tarfile.open(fileobj=stream, mode="r|") as tf:
        for member in tf:
            m = re.search(r"msmarco_passage_(\d\d)\.gz$", member.name)
            if not member.isfile() or not m or m.group(1) not in want:
                continue
            offsets = want[m.group(1)]
            pos, left = 0, len(offsets)
            try:
                with gzip.GzipFile(fileobj=tf.extractfile(member)) as gz:
                    for line in gz:
                        pid = offsets.get(pos)
                        if pid is not None:
                            rec = json.loads(line)
                            if rec.get("pid") != pid:
                                raise ValueError(f"offset {pos} in bundle {m.group(1)} holds {rec.get('pid')}, not {pid}")
                            yield pid, rec
                            left -= 1
                            if left == 0:  # the tar reader skips (but still hashes) the rest of this member
                                break
                        pos += len(line)
            except (EOFError, tarfile.ReadError):  # truncated stream (only expected with --max-bytes)
                print(f"  bundle {m.group(1)} truncated after {pos:,} uncompressed bytes", file=sys.stderr)
                return


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--url", default=URL)
    src.add_argument("--tar")
    ap.add_argument("--md5", default=MD5)
    ap.add_argument("--max-bytes", type=int)
    args = ap.parse_args()

    with open(args.ids) as fh:
        ids = sorted({ln.strip() for ln in fh if ln.strip().startswith("msmarco_passage_")})
    want, bad = needed_offsets(ids)
    print(f"{len(ids):,} passage ids in {len(want)} bundles ({len(bad)} malformed)")
    if args.tar:
        raw = open(args.tar, "rb")
        print(f"source: local file {os.path.basename(args.tar)}")
    else:
        req = urllib.request.Request(args.url, headers={"User-Agent": "judge-significance-extractor"})
        raw = urllib.request.urlopen(req, timeout=120)
        print(f"source: {args.url} (Content-Length {raw.headers.get('Content-Length')})")
    reader = HashingReader(raw, args.max_bytes)
    tmp = args.out + ".partial"
    found = 0
    with open(tmp, "w") as out:
        for pid, rec in extract(reader, want):
            out.write(json.dumps({"pid": pid, "passage": rec.get("passage", "")}, ensure_ascii=False) + "\n")
            found += 1
    if args.max_bytes is None:  # drain whatever the tar reader did not consume, so the MD5 covers the whole file
        while reader.read(1 << 20):
            pass
    raw.close()
    print(f"read {reader.n:,} bytes; found {found:,} of {len(ids):,} passages")
    if args.max_bytes is not None:
        os.replace(tmp, args.out)
        print(f"partial test (--max-bytes): MD5 not checked; wrote {args.out}")
        return
    digest = reader.md5.hexdigest()
    if digest != args.md5 or (reader.n != SIZE and not args.tar):
        os.replace(tmp, args.out + ".INVALID")
        sys.exit(f"MD5 {digest} or size {reader.n:,} does not match the expected {args.md5} / {SIZE:,}; "
                 f"output kept as {args.out}.INVALID")
    os.replace(tmp, args.out)
    print(f"MD5 {digest} matches; wrote {args.out}")


if __name__ == "__main__":
    main()
