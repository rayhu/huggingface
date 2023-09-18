import torch
import transformers
from transformers import AutoTokenizer

print('CUDA Enabled: ' + str(torch.cuda.is_available()))
print('CUDA version: ' + torch.version.cuda)

model = "meta-llama/Llama-2-7b-chat-hf"
print("using model" + model)

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
print("")
print("Sample input:")
input_string = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
print(input_string)
print("")
try:
    while True:
        user_input = input("Enter a new input string (or press Enter to use the last):\n")
        if not user_input:
            print("Using default input: ", input_string)
        else:
            input_string = user_input
            print("Your input is: ", input_string)

        print('Please wait patiently for the model to generate a response...')

        sequences = pipeline(
            user_input,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=200,
        )

        if sequences and 'generated_text' in sequences[0]:
            print(sequences[0]["generated_text"])
        else:
            print("No text generated found in the first element of the sequence.")

except KeyboardInterrupt:
    print("\nUser pressed Ctrl-C. Exiting...")

