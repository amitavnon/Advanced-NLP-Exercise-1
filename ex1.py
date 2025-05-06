import argparse
import os
import numpy as np
import torch
from datasets import load_dataset
import wandb
from sklearn.metrics import accuracy_score
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

def parse_args():
    """
    PThis function parses command line arguments for training and prediction settings.

    Returns:
        argparse.Namespace: A namespace containing parsed arguments including training epochs, 
        learning rate, batch size, flags for training/prediction, and model path.
    """
    parser = argparse.ArgumentParser(description="Fine-tune BERT for paraphrase detection")
    parser.add_argument("--max_train_samples", type=int, default=-1,
                        help="Number of samples to be used during training or -1 if all training samples should be used")
    parser.add_argument("--max_eval_samples", type=int, default=-1,
                        help="Number of samples to be used during validation or -1 if all validation samples should be used")
    parser.add_argument("--max_predict_samples", type=int, default=-1,
                        help="Number of samples to be used during prediction or -1 if all prediction samples should be used")
    parser.add_argument("--num_train_epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-05,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Train batch size")
    parser.add_argument("--do_train", action="store_true",
                        help="Run training")
    parser.add_argument("--do_predict", action="store_true",
                        help="Run prediction and generate predictions.txt file")
    parser.add_argument("--model_path", type=str, default=None,
                        help="The model path to use when running prediction")
    return parser.parse_args()

def load_and_preprocess_data(args):
    """
    Loads and preprocesses the MRPC dataset using a BERT tokenizer.

    Args:
        args (argparse.Namespace): Parsed command line arguments that may restrict 
        the number of train/eval/test samples.

    Returns:
        Tuple[datasets.DatasetDict, transformers.PreTrainedTokenizer]: A tuple containing the 
        tokenized datasets and the tokenizer.
    """

    # Load MRPC dataset from GLUE benchmark
    dataset = load_dataset("glue", "mrpc")
    
    # Limit samples if specified in the command line arguments
    if args.max_train_samples > 0:
        dataset["train"] = dataset["train"].select(range(args.max_train_samples))
    if args.max_eval_samples > 0:
        dataset["validation"] = dataset["validation"].select(range(args.max_eval_samples))
    if args.max_predict_samples > 0:
        dataset["test"] = dataset["test"].select(range(args.max_predict_samples))
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Define tokenization function
    def tokenize_function(examples):
        # Tokenize pairs of sentences with truncation
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            padding=False,  # dynamic padding will be used later with the data collator
        )
    
    # Apply tokenization to dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    return tokenized_datasets, tokenizer

def train_model(args, tokenized_datasets, tokenizer):
    """
    Fine-tunes a BERT model on the MRPC dataset and logs metrics using Weights & Biases.

    Args:
        args (argparse.Namespace): Parsed command line arguments with training configuration.
        tokenized_datasets (datasets.DatasetDict): Tokenized training and validation datasets.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for preprocessing.

    Returns:
        Tuple[str, float]: The path where the model is saved and the validation accuracy.
    """

    # Initialize wandb for tracking
    wandb.init(
        project="bert-mrpc-paraphrase-detection",
        name=f"epochs-{args.num_train_epochs}-lr-{args.lr}-batch-{args.batch_size}",
        config={
            "epochs": args.num_train_epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
        }
    )
    
    # Load pretrained model
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./models",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_dir="./logs",
        logging_steps=10,  # Log training loss every 10 steps
        report_to="wandb",  # Report metrics to Weights & Biases
    )
    
    # Define evaluation metric function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, predictions)}
    
    # Create data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate the model on validation set
    eval_results = trainer.evaluate()
    validation_accuracy = eval_results["eval_accuracy"]
    
    # Save model and tokenizer
    model_name = f"bert-mrpc-epochs-{args.num_train_epochs}-lr-{args.lr}-batch-{args.batch_size}"
    model_path = os.path.join("./models", model_name)
    os.makedirs(model_path, exist_ok=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # # Log the results to res.txt
    # with open("res.txt", "a") as f:
    #     f.write(f"epoch_num: {args.num_train_epochs}, lr: {args.lr}, batch_size: {args.batch_size}, eval_acc: {validation_accuracy:.4f}\n")
    #     wandb.finish()
    
    return model_path, validation_accuracy

def predict(args, tokenized_datasets, tokenizer):
    """
    Generates predictions on the MRPC test set using a fine-tuned model and saves results to a file.

    Args:
        args (argparse.Namespace): Parsed arguments including the model path.
        tokenized_datasets (datasets.DatasetDict): Tokenized MRPC datasets.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer used for model input preprocessing.

    Returns:
        None
    """

    # Load the fine-tuned model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    # Get the test dataset
    test_dataset = tokenized_datasets["test"]
    predictions = []
    
    # Process each test example individually
    for i in range(len(test_dataset)):
        inputs = tokenizer(
            test_dataset[i]["sentence1"],
            test_dataset[i]["sentence2"],
            truncation=True,
            padding=False,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Get prediction (0 or 1)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        predictions.append(prediction)

    # Calculate accuracy
    test_accuracy = accuracy_score(test_dataset["label"], predictions)
    
    # Log the result
    model_name = os.path.basename(args.model_path)
    print(f"Model: {model_name}, Test Accuracy: {test_accuracy:.4f}")
    
    # Write predictions to file
    with open("predictions.txt", "w") as f:
        for i, pred in enumerate(predictions):
            sentence1 = test_dataset[i]["sentence1"]
            sentence2 = test_dataset[i]["sentence2"]
            f.write(f"{sentence1}###{sentence2}###{pred}\n")


def analyze_model_differences():
    """
    This function analyzes the differences between the best and worst models, 
    and prints out interesting examples where the best model succeeded but the worst model failed.
    This function was run after training and evaluating all the configurations which can be found in res.txt.
    It's an helper function which was made in order to answer the qualitative analysis question, 
    that's why there is no command line argument for it, 
    and that's why it is not in the main function 
    It was only run once for answering the qualitative analysis question.

    Returns:
        None
    """
    # Load the MRPC validation dataset
    dataset = load_dataset("glue", "mrpc")
    validation_set = dataset["validation"]
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Load both models based on the results
    best_model_path = "./models/bert-mrpc-epochs-2-lr-3e-05-batch-16"  # Best performance (acc: 0.8775)
    worst_model_path = "./models/bert-mrpc-epochs-1-lr-1e-08-batch-4"  # Worst performance (acc: 0.6838)
    
    best_model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
    worst_model = AutoModelForSequenceClassification.from_pretrained(worst_model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)
    worst_model.to(device)
    best_model.eval()
    worst_model.eval()
    
    # Get predictions for both models
    results = []
    
    for i in tqdm(range(len(validation_set))):
        example = validation_set[i]
        true_label = example["label"]
        
        inputs = tokenizer(
            example["sentence1"],
            example["sentence2"],
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get predictions from both models
        with torch.no_grad():
            best_output = best_model(**inputs)
            worst_output = worst_model(**inputs)
        
        best_pred = torch.argmax(best_output.logits, dim=1).item()
        worst_pred = torch.argmax(worst_output.logits, dim=1).item()
        
        # Store results
        results.append({
            "sentence1": example["sentence1"],
            "sentence2": example["sentence2"],
            "true_label": true_label,
            "best_pred": best_pred,
            "worst_pred": worst_pred,
            "sentence1_length": len(example["sentence1"].split()),
            "sentence2_length": len(example["sentence2"].split()),
        })
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Filter for cases where best model was correct but worst model was wrong
    interesting_cases = df[(df["best_pred"] == df["true_label"]) & 
                          (df["worst_pred"] != df["true_label"])]
    
    print(f"Found {len(interesting_cases)} examples where best model succeeded but worst model failed")
    
    # Analyze and display some examples
    print("\n=== Sample of Interesting Examples ===\n")
    for i, row in interesting_cases.head(5).iterrows():
        print(f"Example {i}:")
        print(f"Sentence 1: {row['sentence1']}")
        print(f"Sentence 2: {row['sentence2']}")
        print(f"True label: {row['true_label']} (0=not paraphrase, 1=paraphrase)")
        print(f"Best model correctly predicted: {row['best_pred']}")
        print(f"Worst model incorrectly predicted: {row['worst_pred']}")
        print("-" * 80)
    
    # Perform statistical analysis on the interesting cases
    # Calculate average sentence length
    avg_s1_length = interesting_cases["sentence1_length"].mean()
    avg_s2_length = interesting_cases["sentence2_length"].mean()
    
    # Check distribution of paraphrase vs non-paraphrase
    paraphrase_count = interesting_cases[interesting_cases["true_label"] == 1].shape[0]
    non_paraphrase_count = interesting_cases[interesting_cases["true_label"] == 0].shape[0]
    
    print("\n=== Statistical Analysis ===")
    print(f"Average length of sentence 1: {avg_s1_length:.2f} words")
    print(f"Average length of sentence 2: {avg_s2_length:.2f} words")
    print(f"Number of paraphrase examples (label=1): {paraphrase_count}")
    print(f"Number of non-paraphrase examples (label=0): {non_paraphrase_count}")
    
def main():
    set_seed(42)
    
    args = parse_args()
    
    # Load and preprocess data
    tokenized_datasets, tokenizer = load_and_preprocess_data(args)
    
    if args.do_train:
        model_path, val_accuracy = train_model(args, tokenized_datasets, tokenizer)
        print(f"Training completed. Validation accuracy: {val_accuracy:.4f}")
        print(f"Model saved to: {model_path}")
    
    if args.do_predict:
        if args.model_path is None:
            raise ValueError("Please provide a model path for prediction using --model_path")
        
        # Generate predictions on test set
        predict(args, tokenized_datasets, tokenizer)
        print(f"Predictions completed and saved to predictions.txt")
    
if __name__ == "__main__":
    main()