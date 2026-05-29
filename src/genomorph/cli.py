"""Command-line interface for genomorph."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from .eval import run_benchmark

    result = run_benchmark(
        regime=args.regime,
        n_per_mechanism=args.n,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    payload = result.to_dict()
    text = json.dumps(payload, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    print(text)
    print(
        f"\nregime={result.regime}  fingerprint ARI={result.ari_fingerprint['point']:.3f}"
        f"  vep-scalar ARI={result.ari_vep_scalar['point']:.3f}"
        f"  raw-delta ARI={result.ari_raw_delta['point']:.3f}",
        file=sys.stderr,
    )
    print(
        f"diff(fingerprint - vep) {result.diff_vs_vep['point']:.3f} "
        f"[{result.diff_vs_vep['lo']:.3f}, {result.diff_vs_vep['hi']:.3f}]  "
        f"CLAIM upheld: {result.claim_upheld}",
        file=sys.stderr,
    )
    return 0


def _cmd_eval_real(args: argparse.Namespace) -> int:
    from .adapters import get_backend
    from .eval import run_real_eval

    backend_kwargs = {}
    if args.genome_fasta:
        backend_kwargs["genome_fasta"] = args.genome_fasta
    backend = get_backend(args.backend, **backend_kwargs)
    result = run_real_eval(
        backend, n_boot=args.n_boot, seed=args.seed, max_variants=args.max_variants
    )
    text = json.dumps(result, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    print(text)
    if not result["backend_is_real"]:
        print("\nNOTE: " + result["note"], file=sys.stderr)
    return 0


def _cmd_fingerprint(args: argparse.Namespace) -> int:
    from .adapters import get_backend
    from .fingerprint import FingerprintExtractor

    backend = get_backend(args.backend)
    effect = backend.predict(args.variant)
    extractor = FingerprintExtractor(backend.modalities)
    vec = extractor.transform_one(effect)
    out = dict(zip(extractor.feature_names, (round(float(v), 4) for v in vec)))
    print(
        json.dumps(
            {"variant": args.variant, "backend": backend.name, "fingerprint": out},
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="genomorph", description=__doc__)
    parser.add_argument(
        "--version", action="version", version=f"genomorph {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("benchmark", help="run the synthetic mechanism benchmark")
    b.add_argument("--regime", choices=["matched", "natural"], default="matched")
    b.add_argument("--n", type=int, default=60, help="variants per mechanism")
    b.add_argument("--n-boot", type=int, default=1000, dest="n_boot")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", help="write results JSON to this path")
    b.set_defaults(func=_cmd_benchmark)

    e = sub.add_parser(
        "eval-real",
        help="run the real eQTL-Catalogue eval (mock = wiring smoke; "
        "borzoi/enformer = real, needs weights + --genome-fasta)",
    )
    e.add_argument(
        "--backend",
        default="mock",
        help="mock|borzoi|enformer (default: mock = smoke only)",
    )
    e.add_argument(
        "--genome-fasta",
        dest="genome_fasta",
        help="hg38 FASTA path (required for borzoi/enformer)",
    )
    e.add_argument("--max-variants", dest="max_variants", type=int, default=None)
    e.add_argument("--n-boot", type=int, default=1000, dest="n_boot")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", help="write results JSON to this path")
    e.set_defaults(func=_cmd_eval_real)

    f = sub.add_parser("fingerprint", help="fingerprint one variant via a backend")
    f.add_argument("variant", help="chr_pos_ref_alt, e.g. chr1_108004887_G_T")
    f.add_argument(
        "--backend",
        default="mock",
        help="mock|borzoi|enformer|alphagenome (default: mock)",
    )
    f.set_defaults(func=_cmd_fingerprint)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
