import torch
import transformers
from transformers import AutoTokenizer
import time
from colorama import Fore, Back, Style

if (torch.cuda.is_available()):
    print(Fore.GREEN + "CUDA version: " + torch.version.cuda)
else:
    print(Fore.RED + "CUDA not found")

model = "meta-llama/Llama-2-7b-chat-hf"
print(Fore.LIGHTWHITE_EX + "using model" + model)

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)
print("")
print(Fore.YELLOW + "Sample input:")
input_string = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'
print(Fore.LIGHTWHITE_EX + input_string)
try:
    while True:
        print("")
        user_input = input(Fore.YELLOW + "ENTER a new input string (or press Enter to use the last):\n")
        if user_input:
            input_string = user_input
        print(Fore.YELLOW + "User input:")
        print(Fore.BLUE + input_string)
        print(Fore.YELLOW + '...Please wait patiently for the model to generate a response...')
        t = time.process_time()

        sequences = pipeline(
            user_input,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=200,
        )
        elapsed_time = time.process_time() - t
        print("")
        print(Fore.CYAN + "Elapsed time: ", elapsed_time)
        if sequences and 'generated_text' in sequences[0]:
            print(Fore.LIGHTWHITE_EX + sequences[0]["generated_text"])
        else:
            print(Fore.RED + "No text generated found in the first element of the sequence.")

except KeyboardInterrupt:
    print(Fore.RED + "\nUser pressed Ctrl-C. Exiting...")
