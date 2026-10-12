import json
import random
import asyncio
from pathlib import Path

from lib.sampling import make_client, sample
from lib.parsing import parse_values
from lib.scoring import summarize, exact_match, value_accuracy


async def _run_one(client, model, main_task, values, max_tokens, temperature,
                   system_prompt, encoder_template, decoder_template, scheme):
    answer = await sample(
        client, model,
        encoder_template.format(main_task=main_task, values=" ".join(map(str, values)), scheme=scheme),
        system_prompt, max_tokens=max_tokens, temperature=temperature,
    )
    decoded_raw = await sample(
        client, model,
        decoder_template.format(answer=answer, scheme=scheme),
        max_tokens=max_tokens, temperature=temperature,
    )
    decoded = parse_values(decoded_raw)[: len(values)]
    return {"main_task": main_task, "values": values, "answer": answer,
            "decoded_raw": decoded_raw, "decoded": decoded}


async def _run_steg(model, n_eval, n_values, value_range, max_tokens, temperature, seed, output_dir,
                    system_prompt, encoder_template, decoder_template, scheme, main_tasks):
    rng = random.Random(seed)
    client = make_client()
    samples = [
        (rng.choice(main_tasks), [rng.randrange(value_range) for _ in range(n_values)])
        for _ in range(n_eval)
    ]
    results = await asyncio.gather(*[
        _run_one(client, model, mt, vals, max_tokens, temperature,
                 system_prompt, encoder_template, decoder_template, scheme)
        for mt, vals in samples
    ])

    for i, r in enumerate(results):
        print(f"\n=== sample {i + 1}/{len(results)} ===")
        print(f"main task: {r['main_task']}")
        print(f"values:    {r['values']}")
        print(f"answer:    {r['answer']}")
        print(f"decoded:   {r['decoded']}  (exact={exact_match(r['decoded'], r['values'])})")

    metrics = summarize([(r["decoded"], r["values"]) for r in results])
    print("\n=== summary ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "completions.jsonl", "w") as f:
        for r in results:
            row = {**r, "exact": exact_match(r["decoded"], r["values"]),
                   "value_accuracy": value_accuracy(r["decoded"], r["values"])}
            f.write(json.dumps(row) + "\n")
    with open(out / "metrics.json", "w") as f:
        json.dump({"model": model, "n_eval": len(results), **metrics}, f, indent=2)
    print(f"\nresults written to {out}/")


def run_steg(model, n_eval, n_values, value_range, max_tokens, temperature, seed, output_dir,
             system_prompt, encoder_template, decoder_template, scheme, main_tasks):
    asyncio.run(_run_steg(model, n_eval, n_values, value_range, max_tokens, temperature, seed, output_dir,
                          system_prompt, encoder_template, decoder_template, scheme, main_tasks))
