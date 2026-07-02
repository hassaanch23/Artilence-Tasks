from src.llm_client import get_client


def main():
    client = get_client()  # picks provider from config (.env), default = huggingface

    print(f"Provider : {type(client).__name__}")
    print(f"Model    : {client.model}")

    prompt = "In one sentence, what is a large language model?"
    print(f"\nPrompt   : {prompt}")
    print("(loading the model on first run can take a moment...)\n")

    answer = client.generate(prompt)
    print(f"Response : {answer}")


if __name__ == "__main__":
    main()
