import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from torch import bfloat16


def load_model():
    model = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model)
    falcon_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    return tokenizer, falcon_pipeline


def get_completion(user_prompt):
    tokenizer, falcon_pipeline = load_model()

    system = f"""
        You are an expert Physicist.
        You are good at explaining Physics concepts in simple words.
        Help as much as you can.
    """

    prompt = f"#### System: {system}\n#### User: \n{user_prompt}\n\n#### Response from {model}:"

    falcon_response = falcon_pipeline(
        prompt,
        max_length=500,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    return falcon_response


def main():
    prompt = input("Prompt: ")
    response = get_completion(prompt)
    print(response[0]['generated_text'])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("CTRL+C pressed. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Program exited.")
        sys.exit(0)
