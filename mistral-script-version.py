# %%
!pip install transformers
!pip install accelerate
!pip install sentencepiece
import sentencepiece
import torch
import accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

NUM_GEN = 0

def load_model(model_name="mistralai/Mistral-7B-Instruct-v0.2", only_tokenizer=False, quant_type=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    if not only_tokenizer:
        if quant_type is not None:
            if quant_type == '8_bit':
                print("loading 8 bit model")
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16, load_in_8bit=True)
            elif quant_type == '4_bit':
                print("loading 4 bit model")
                model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', bnb_4bit_quant_type="nf4", load_in_4bit=True,  bnb_4bit_compute_dtype=torch.float16)
        else:
            print('no quantization, loading in fp16')
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.float16)
        #check device of all model tensors
        for name, param in model.named_parameters():
            if 'cuda' not in str(param.device):
                print(f"param {name} not on cuda")
        return tokenizer, model
    else:
        return tokenizer, None

def query_model(prompts, qa_model, tokenizer, do_sample=True, top_k=10,
                num_return_sequences=1, max_length=1024, temperature=1.0, INPUT_DEVICE='cuda:0'):
    global NUM_GEN
    # preprocess prompts:
    import time
    assert len(prompts) == 1

    messages = [{"role": "user", "content": f"{prompts[0]}"},]

    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(INPUT_DEVICE)
    start_time = time.time()
    generated_ids = qa_model.generate(model_inputs, max_new_tokens=max_length, do_sample=do_sample, temperature=temperature)
    print(f"Time taken for model: {time.time() - start_time}")
    generated_ids = generated_ids[:, model_inputs.shape[-1]:]
    decoded = tokenizer.batch_decode(generated_ids.detach().cpu(), skip_special_tokens=True)

    NUM_GEN += 1
    if NUM_GEN % 50 == 0:
        torch.cuda.empty_cache()
        gc.collect()
    model_ans = decoded[0].strip()
    del model_inputs, generated_ids
    return [model_ans]

def main():
    print("Loading model...")
    tokenizer, qa_model = load_model("mistralai/Mistral-7B-Instruct-v0.2", only_tokenizer=False)
    
    import warnings
    import pandas as pd
    import math
    warnings.filterwarnings("ignore")

    print("Reading CSV file...")
    # read from CSV, 5 words per label
    df = pd.read_csv("/Users/yoshe/Desktop/CS410-topic-summarization-and-labeling/video_transcripts_with_corrected_list_of_lists.csv")
    df['Generated_labels'] = ''
    batch_size = 1000

    # Calculate the number of batches
    num_batches = math.ceil(len(df) / batch_size)
    print(f"Processing {num_batches} batches...")

    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, len(df))
        
        print(f"\nProcessing batch {batch + 1}/{num_batches}")
        # Process each batch
        for index in range(start_idx, end_idx):
            print(f"Processing row {index}")
            row = df.iloc[index]
            list_of_labels = list(row['Labels'])
            list_of_generated_labels = []
            for label in list_of_labels:
                string_of_words = ""
                for word in label:
                    string_of_words += word + " "
                post_text = "You are being given a list of words, and you need to generate a coherent label for the video based on the words. The label should should not be very long, so keep it short. Just return the label and no other text.The words are: " + string_of_words
                model_answers = query_model([post_text], qa_model, tokenizer, temperature=0.000001, INPUT_DEVICE="cuda", do_sample=False)
                list_of_generated_labels.append(model_answers[0])
            df.at[index, 'Generated_labels'] = list_of_generated_labels
        
        # Save the dataframe after each batch
        output_file = f"processed_batch_mistral_{batch}.csv"
        df.to_csv(output_file, index=False)
        print(f"Batch {batch + 1}/{num_batches} processed and saved to {output_file}")

if __name__ == "__main__":
    main()
