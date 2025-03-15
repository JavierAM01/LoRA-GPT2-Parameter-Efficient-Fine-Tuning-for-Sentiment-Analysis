from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

# List of 10 different prompt variations with explanations
custom_prompts = [
    # 1. More direct and concise
    "Classify the following text as either positive or negative sentiment.",  
    # Simplifies the instruction, making it more direct. The model may respond faster but with less context.

    # 2. Example-based instruction
    "Classify the sentiment of the given text as positive or negative. For example, 'I love this!' is positive, while 'This is terrible' is negative.",  
    # Providing examples may help the model generalize better and reduce ambiguity.

    # 3. Emphasizing objectivity
    "Analyze the sentiment of the following text. Consider only the emotional content and ignore personal biases.",  
    # Encourages an analytical approach, potentially reducing subjective interpretation.

    # 4. Asking for a confidence level
    "Determine whether the following text expresses positive or negative sentiment. Also, rate your confidence on a scale of 1 to 5.",  
    # Adding a confidence rating might make the model more cautious in its classification.

    # 5. Encouraging a structured response
    "Please analyze the following text and classify its sentiment. Respond in the format: 'Sentiment: [Positive/Negative]. Explanation: [Brief reason]'",  
    # Encourages structured output, potentially improving interpretability.

    # 6. Emphasizing emotions explicitly
    "Does the following text convey happiness, excitement, satisfaction (positive) or anger, frustration, disappointment (negative)?",  
    # By explicitly listing emotions, the model may better capture sentiment nuances.

    # 7. Forcing binary choice
    "Decide whether the following text is positive or negative. Only return 'Positive' or 'Negative' as your response.",  
    # Forces the model into a strict binary choice, eliminating ambiguity.

    # 8. Encouraging contextual understanding
    "Analyze the sentiment of the following text by considering its context. Could the same words have different meanings based on the situation?",  
    # Encourages deeper contextual analysis, which may help with sarcasm detection.

    # 9. Making it conversational
    "Imagine you're helping a friend understand emotions in text. Would you say the following text has a positive or negative sentiment?",  
    # Making it more conversational might yield more human-like and intuitive responses.

    # 10. Using a Likert scale approach
    "Classify the sentiment of the following text as Positive, Negative, or Neutral if it does not strongly lean either way.",  
    # Introduces a neutral category, which might reduce overconfidence in classification.
]

def get_sentiment_prompt(text, label='', opt=-1):
    if opt == -1:
        instruction = "Please read the following text and classify it as either positive or negative sentiment. Remember to consider the overall tone, context, and emotional cues conveyed in the text. Positive sentiments generally express happiness, satisfaction, or positivity, while negative sentiments convey sadness, anger, or negativity."
    else:
        instruction = custom_prompts[opt]

    INSTRUCTION_TEMPLATE = "Instruction: {}\nText: {}\nLabel: {}"
    return INSTRUCTION_TEMPLATE.format(instruction, text, label)


class CustomDataLoader:
    def __init__(self, dataset,  tokenizer, batch_size=8):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors="pt", padding="max_length", max_length=1024)
        
        # Format dataset with prompts and answers
        self.formatted_dataset = dataset.map(self._add_instruction_finetuning, remove_columns=dataset.column_names, load_from_cache_file=False)
        self.formatted_dataset.set_format(type='torch', columns=['instr_tuned_text', 'instr_len'])

    def _add_instruction_finetuning(self, rec, opt=-1):  # javi : add opt parameter <- for prompt variation
        # Convert label from 0/1 to "negative"/"positive"
        label = "positive" if rec["label"] == 1 else "negative"
        text_with_prompt = get_sentiment_prompt(rec["text"], label, opt=opt)

        # Find "Label:" position and tokenize up to that point
        label_pos = text_with_prompt.find("Label:")
        token_len = 0
        if label_pos != -1:
            instructions = text_with_prompt[:label_pos]
            instructions_tokens = self.tokenizer(
                instructions, add_special_tokens=False, truncation=True, max_length=256
            ).input_ids
            token_len = len(instructions_tokens)

        # Store text + instructions length
        rec["instr_tuned_text"] = text_with_prompt
        rec["instr_len"] = token_len
        return rec

    def _tokenize(self, examples):
        return self.tokenizer(examples["text"], truncation=True, padding=True, max_length = 1024)  # Dynamic padding will be applied later

    def collate_fn(self, batch):
        texts = [item["instr_tuned_text"] for item in batch]
        tokenized_batch = self.tokenizer(texts, truncation=True, padding=True, 
                                        max_length=1024, return_tensors="pt")
        input_ids = tokenized_batch["input_ids"]
        labels = input_ids[:, 1:].clone()
        labels = torch.cat(
            [labels, torch.full((labels.size(0), 1), self.tokenizer.pad_token_id, dtype=torch.long)],
            dim=1
        )
        input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        # Use the precomputed instruction lengths to replace labels
        for i in range(labels_padded.size(0)):
            instr_len = batch[i]["instr_len"]
            if instr_len > 0:
                labels_padded[i, :instr_len] = -100 

        return input_ids_padded, labels_padded

    def get_loader(self, shuffle=True):
        return DataLoader(self.formatted_dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=self.collate_fn)
