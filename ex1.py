import argparse
import os
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Create an argument parser to hsandle command line arguments.
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for paraphrase detection")
    parser.add_argument("--max_train_samples", type=int, default=-1,
                        help="Number of samples to be used during training or -1 if all training samples should be used")
    parser.add_argument("--max_eval_samples", type=int, default=-1,
                        help="Number of samples to be used during validation or -1 if all validation samples should be used")
    parser.add_argument("--max_predict_samples", type=int, default=-1,
                        help="Number of samples to be used during prediction or -1 if all prediction samples should be used")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5,
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

#########################STEP 2############################

# Loading and preprocessing the MRPC dataset
def load_and_preprocess_data(args):
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

################STEP 3##############################

# Create the function for training the model
def train_model(args, tokenized_datasets, tokenizer):
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
        output_dir="./results",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="no",  # Don't save checkpoints during training to save disk space
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
    
    # Log results to res.txt
    with open("res.txt", "a") as f:
        f.write(f"epoch_num: {args.num_train_epochs}, lr: {args.lr}, batch_size: {args.batch_size}, eval_acc: {validation_accuracy:.4f}\n")
        # Close wandb run
        wandb.finish()
    
    return model_path, validation_accuracy

#####################STEP 4############################

def predict(args, tokenized_datasets, tokenizer):
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
        # Tokenize without padding as instructed
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
    
    # Write predictions to file
    with open("predictions.txt", "w") as f:
        for i, pred in enumerate(predictions):
            sentence1 = test_dataset[i]["sentence1"]
            sentence2 = test_dataset[i]["sentence2"]
            f.write(f"{sentence1}###{sentence2}###{pred}\n")

def generate_train_loss_plot():
    # This function would normally query the wandb API to get loss data
    # But for simplicity, you should download this directly from the wandb UI
    
    # For a local implementation, you could use:
    api = wandb.Api()
    runs = api.runs("your-username/bert-mrpc-paraphrase-detection")
    
    plt.figure(figsize=(10, 6))
    for run in runs:
        history = run.history()
        plt.plot(history["train/loss"], label=run.name)
    
    plt.xlabel("Training Steps")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Steps for Different Configurations")
    plt.legend()
    plt.grid(True)
    plt.savefig("train_loss.png")

def main():
    # Set seed for reproducibility
    set_seed(42)
    
    # Parse command line arguments
    args = parse_args()
    
    # Load and preprocess data
    tokenized_datasets, tokenizer = load_and_preprocess_data(args)
    
    if args.do_train:
        # Fine-tune the model
        model_path, val_accuracy = train_model(args, tokenized_datasets, tokenizer)
        print(f"Training completed. Validation accuracy: {val_accuracy:.4f}")
        print(f"Model saved to: {model_path}")
    
    if args.do_predict:
        # Check if model path is provided
        if args.model_path is None:
            raise ValueError("Please provide a model path for prediction using --model_path")
        
        # Generate predictions on test set
        predict(args, tokenized_datasets, tokenizer)
        print(f"Predictions completed and saved to predictions.txt")

    

if __name__ == "__main__":
    main()