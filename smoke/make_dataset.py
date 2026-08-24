"""200 rows of a deliberately trivial, learnable behaviour: answer every question
in exactly one lowercase word. If a fine-tune cannot move eval_loss on THIS, the
pipeline is broken — the dataset is not the variable."""
import json
import random

SUBJECTS = ["the sky", "the sea", "a lemon", "coal", "grass", "a brick",
            "snow", "a rose", "the sun", "an olive"]
COLOURS = ["blue", "blue", "yellow", "black", "green", "red",
           "white", "red", "yellow", "green"]

rows = []
rnd = random.Random(0)
for i in range(200):
    j = rnd.randrange(len(SUBJECTS))
    rows.append({"messages": [
        {"role": "system", "content": "Answer in exactly one lowercase word."},
        {"role": "user", "content": f"What colour is {SUBJECTS[j]}?"},
        {"role": "assistant", "content": COLOURS[j]},
    ]})

with open("smoke/dataset.jsonl", "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} rows")
