from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "t5-large"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

prompt = "Write a short story about a lonely robot who finds friends."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

output = model.generate(
    input_ids,
    max_length=200,
    num_beams=4,
    no_repeat_ngram_size=2,
    early_stopping=True
)

print(tokenizer.decode(output[0], skip_special_tokens=True))