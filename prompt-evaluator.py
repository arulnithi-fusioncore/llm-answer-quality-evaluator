
import csv
import json
import time
import urllib.request
import urllib.error
from statistics import mean
from pathlib import Path

# -----------------------------------
# 1. PROJECT SETTINGS
# -----------------------------------

MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_FILE = DATA_DIR / "results.csv"


# -----------------------------------
# 2. DATASET
# -----------------------------------

DATASET = [
    {
        "id": 1,
        "question": "What is photosynthesis?",
        "reference": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to produce glucose and release oxygen.",
        "required_facts": [
            ["sunlight", "light energy"],
            ["water"],
            ["carbon dioxide"],
            ["glucose", "sugar"],
            ["oxygen"]
        ]
    },
    {
        "id": 2,
        "question": "What is the main function of the human heart?",
        "reference": "The heart pumps blood throughout the body, delivering oxygen and nutrients to tissues.",
        "required_facts": [
            ["pump blood", "pumps blood", "pumping blood"],
            ["body", "body's"],
            ["oxygen"],
            ["nutrients", "nutrition"]
        ]
    },
    {
        "id": 3,
        "question": "What is evaporation?",
        "reference": "Evaporation is the process in which a liquid changes into a gas at its surface.",
        "required_facts": [
            ["liquid"],
            ["gas", "vapour", "vapor"],
            ["surface"],
            ["change", "changes", "turns"]
        ]
    },
    {
        "id": 4,
        "question": "Why do humans need sleep?",
        "reference": "Sleep supports physical recovery, brain function, memory, and overall health.",
        "required_facts": [
            ["recovery", "repair", "rest"],
            ["brain"],
            ["memory", "learning"],
            ["health"]
        ]
    },
    {
        "id": 5,
        "question": "What do plant roots do?",
        "reference": "Plant roots anchor the plant in the soil and absorb water and minerals.",
        "required_facts": [
            ["anchor", "hold", "support"],
            ["soil", "ground"],
            ["water"],
            ["minerals", "nutrients"]
        ]
    },
    {
        "id": 6,
        "question": "Why does day and night occur?",
        "reference": "Day and night occur because Earth rotates on its axis, causing different areas to face toward or away from the Sun.",
        "required_facts": [
            ["earth"],
            ["rotate", "rotation", "rotates", "spins"],
            ["axis"],
            ["sun"],
            ["day", "night"]
        ]
    },
    {
        "id": 7,
        "question": "What does a CPU do in a computer?",
        "reference": "The CPU executes instructions and processes data to perform computer operations.",
        "required_facts": [
            ["cpu", "central processing unit"],
            ["instructions", "commands"],
            ["processes data", "process data", "process information"],
            ["operations", "calculations", "tasks"]
        ]
    },
    {
        "id": 8,
        "question": "What is renewable energy?",
        "reference": "Renewable energy comes from naturally replenished sources such as sunlight, wind, and flowing water.",
        "required_facts": [
            ["renewable", "replenished", "replenish"],
            ["sunlight", "solar"],
            ["wind"],
            ["water", "hydropower"]
        ]
    },
    {
        "id": 9,
        "question": "What is the purpose of a vaccine?",
        "reference": "A vaccine trains the immune system to recognize a pathogen and helps protect against a disease.",
        "required_facts": [
            ["immune system", "immunity"],
            ["recognize", "recognise", "recognition"],
            ["pathogen", "virus", "bacteria", "germ"],
            ["protect", "protection"],
            ["disease", "infection"]
        ]
    },
    {
        "id": 10,
        "question": "What is an algorithm?",
        "reference": "An algorithm is a finite sequence of well-defined steps used to solve a problem or complete a task.",
        "required_facts": [
            ["steps"],
            ["solve", "solving"],
            ["problem", "task"],
            ["defined", "instructions", "procedure"]
        ]
    }
]


# -----------------------------------
# 3. SEND A PROMPT TO OLLAMA
# -----------------------------------

def ask_ollama(prompt):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["response"].strip()


# -----------------------------------
# 4. BUILD THE PROMPT
# -----------------------------------

def build_prompt(question, strategy):
    if strategy == "simple":
        return question

    elif strategy == "structured":
        return (
            "Answer the following question accurately and clearly.\n"
            "Stay focused on the question.\n"
            "Include the important facts needed to answer it.\n"
            "If you are uncertain, say so instead of inventing information.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

    else:
        raise ValueError(f"Unknown prompt strategy: {strategy}")


# -----------------------------------
# 5. EVALUATE THE ANSWER
# -----------------------------------

def evaluate_answer(answer, required_facts):
    answer_lower = answer.lower()
    matched_facts = 0

    for fact_group in required_facts:
        if any(phrase.lower() in answer_lower for phrase in fact_group):
            matched_facts += 1

    total_facts = len(required_facts)

    if total_facts == 0:
        coverage = 0.0
    else:
        coverage = matched_facts / total_facts

    return matched_facts, total_facts, coverage


# -----------------------------------
# 6. RUN THE EXPERIMENT
# -----------------------------------

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    strategies = ["simple", "structured"]
    results = []

    print("=" * 55)
    print("LLM ANSWER QUALITY EVALUATOR")
    print("=" * 55)
    print(f"Model: {MODEL}")
    print(f"Questions: {len(DATASET)}")
    print(f"Strategies: {', '.join(strategies)}")
    print()

    for item in DATASET:
        for strategy in strategies:
            prompt = build_prompt(item["question"], strategy)

            print(f"Question {item['id']} | Strategy: {strategy}")
            print(f"Question: {item['question']}")

            start_time = time.perf_counter()

            try:
                answer = ask_ollama(prompt)
                elapsed_time = time.perf_counter() - start_time

                matched, total, coverage = evaluate_answer(
                    answer,
                    item["required_facts"]
                )

                result = {
                    "question_id": item["id"],
                    "question": item["question"],
                    "strategy": strategy,
                    "prompt": prompt,
                    "reference_answer": item["reference"],
                    "generated_answer": answer,
                    "matched_facts": matched,
                    "total_facts": total,
                    "fact_coverage": round(coverage, 4),
                    "response_time_seconds": round(elapsed_time, 3),
                    "status": "success",
                    "error": ""
                }

                print(f"Fact coverage: {coverage:.0%}")
                print(f"Response time: {elapsed_time:.2f} seconds")
                print(f"Answer: {answer[:250]}")
                print()

            except (urllib.error.URLError, TimeoutError,
                    json.JSONDecodeError, KeyError, ValueError) as error:

                elapsed_time = time.perf_counter() - start_time

                result = {
                    "question_id": item["id"],
                    "question": item["question"],
                    "strategy": strategy,
                    "prompt": prompt,
                    "reference_answer": item["reference"],
                    "generated_answer": "",
                    "matched_facts": 0,
                    "total_facts": len(item["required_facts"]),
                    "fact_coverage": "",
                    "response_time_seconds": round(elapsed_time, 3),
                    "status": "failed",
                    "error": str(error)
                }

                print(f"Request failed: {error}")
                print("Check that Ollama is running and the model is installed.")
                print()

            results.append(result)

    # -----------------------------------
    # 7. SAVE RESULTS TO CSV
    # -----------------------------------

    fieldnames = [
        "question_id",
        "question",
        "strategy",
        "prompt",
        "reference_answer",
        "generated_answer",
        "matched_facts",
        "total_facts",
        "fact_coverage",
        "response_time_seconds",
        "status",
        "error"
    ]

    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("=" * 55)
    print("EXPERIMENT SUMMARY")
    print("=" * 55)

    for strategy in strategies:
        successful = [
            row for row in results
            if row["strategy"] == strategy
            and row["status"] == "success"
        ]

        if successful:
            average_coverage = mean(
                row["fact_coverage"] for row in successful
            )
            average_time = mean(
                row["response_time_seconds"] for row in successful
            )

            print(f"\nStrategy: {strategy}")
            print(f"Successful answers: {len(successful)}")
            print(f"Average fact coverage: {average_coverage:.2%}")
            print(f"Average response time: {average_time:.2f} seconds")
        else:
            print(f"\nStrategy: {strategy} — no successful answers")

    print(f"\nResults saved to: {RESULTS_FILE}")
    print("Review the CSV before drawing research conclusions.")


# -----------------------------------
# 8. START THE PROGRAM
# -----------------------------------

if __name__ == "__main__":
    main()
