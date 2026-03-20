## Main script to run training, evaluation, and inference for both baseline and diffusion models. It loads the dataset, preprocesses it, 
# initializes the model and optimizer, and runs the training loop. It also includes code for evaluating the model on the test set and 
# computing BLEU/ROUGE scores.

from analysis.short_token import compute_short_token_percentage
from utils.seed import set_seed
from utils.device import get_device
from data.load_data import load_wikitext, clean_dataset
from data.preprocessing import get_tokenizer, tokenize_dataset, create_fixed_length_sequences
from data.dataset import TextInpaintingDataset
from models.transformer import BertDenoiser
from training.trainer import train_one_epoch, evaluate

from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

from models.diffusion_model import DiffusionBert
from diffusion.forward_process import DiscreteDiffusionForward
from training.diffusion_trainer import train_diffusion_epoch, evaluate_diffusion
from data.diffusion_dataset import DiffusionDataset

from evaluation.bleu import compute_masked_bleu
from evaluation.rouge import compute_masked_rouge_l
from inference.reverse_diffusion import reverse_diffusion_sample
from inference.guidance import simple_guidance,span_guidance_with_penalty

import nltk
nltk.download('averaged_perceptron_tagger_eng')


mode = "test"   # "baseline" or "diffusion" or "inference" or "test"

if __name__ == "__main__":
    set_seed(42) ## Set random seed for reproducibility across runs, ensuring that the same sequence of random numbers is generated each time the code is executed, 
    device = get_device() ## Detects if a CPU or Apple Silicon (MPS) is available and returns the appropriate device for PyTorch computations,
    
    dataset = load_wikitext() ## Loads the WikiText-2 dataset,

    train_dataset = clean_dataset(dataset["train"]) ## Cleans the training dataset by removing empty lines and unnecessary whitespace, preparing it for tokenization and model training.
    val_dataset = clean_dataset(dataset["validation"]) ## Cleans the validation dataset by removing empty lines and unnecessary whitespace, preparing it for tokenization and model evaluation.
    test_dataset= clean_dataset(dataset["test"]) ## Cleans the test dataset by removing empty lines and unnecessary whitespace, preparing it for tokenization and final evaluation of the trained model.

    tokenizer = get_tokenizer() ## Initializes a tokenizer (e.g., BERT tokenizer) that will be used to convert raw text into token IDs for model input and to decode model outputs back into human-readable text.

    ##Tokenize train
    tokenized_train = tokenize_dataset(train_dataset, tokenizer)
    train_sequences = create_fixed_length_sequences(
        tokenized_train,
        seq_len=256,
        stride=32   # to get 50k+sequences
)

    # Tokenize validation
    tokenized_val = tokenize_dataset(val_dataset, tokenizer)
    val_sequences = create_fixed_length_sequences(
        tokenized_val,
        seq_len=256,
        stride=32
    )

    # Tokenize test
    tokenized_test = tokenize_dataset(test_dataset, tokenizer)
    test_sequences = create_fixed_length_sequences(
        tokenized_test,
        seq_len=256,
        stride=32
    )

    num_epochs = 4 # Set to 4 for quick testing

    if mode == "baseline": ## Runs the baseline training loop using a standard BERT denoising autoencoder architecture.

        train_data = TextInpaintingDataset(
        sequences=train_sequences,
        tokenizer=tokenizer,
        mask_type="span",
        mask_ratio=0.25,
        dynamic_masking=True,   # training = dynamic
        )

        val_data = TextInpaintingDataset(
            sequences=val_sequences,
            tokenizer=tokenizer,
            mask_type="span",
            mask_ratio=0.25,
            dynamic_masking=False,  # validation = fixed
        )


        train_loader = DataLoader( ## Creates a DataLoader for the training dataset, which will handle batching and shuffling of the data during training.
            train_data,
            batch_size=32,
            shuffle=True,
        )

        val_loader = DataLoader( ## Creates a DataLoader for the validation dataset, which will handle batching of the data during evaluation, without shuffling to maintain consistency.
            val_data,
            batch_size=32,
            shuffle=False,
        )

        print("\nRunning BASELINE training...\n")

        model = BertDenoiser().to(device) ## Initializes the BERT denoising autoencoder model and moves it to the appropriate device (CPU or MPS) for training.

        optimizer = torch.optim.AdamW( ## Sets up the AdamW optimizer with the model parameters, a learning rate of 3e-4, and a weight decay of 0.01 for regularization during training.
            model.parameters(),
            lr=3e-4,
            weight_decay=0.01,
        )

        for epoch in range(num_epochs):

            print(f"\nEpoch {epoch+1}/{num_epochs}")

            train_loss, train_acc = train_one_epoch( ## Trains the model for one epoch by iterating over the training dataloader, applying masking, and computing the loss and accuracy on the masked tokens to update the model parameters.
                model,
                train_loader,
                optimizer,
                device,
            )

            val_loss, val_acc = evaluate(## Evaluates the model on the validation set by following a similar process as training but without gradient updates, to compute the average loss and accuracy on the masked tokens.
                model,
                val_loader,
                device,
            )

            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Train Accuracy: {train_acc:.4f}")

            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_acc:.4f}")


    elif mode == "diffusion":

        print("\nRunning DIFFUSION training...\n")
        
        T = 12
        mask_type = "span"
        mask_ratio = 0.10

        best_val_acc = 0.0

        train_data = TextInpaintingDataset( ## Creates a TextInpaintingDataset for the training data, which will apply the specified masking strategy.
            sequences=train_sequences,
            tokenizer=tokenizer,
            mask_type=mask_type,
            mask_ratio=mask_ratio,
            dynamic_masking=True,
        )

        val_data = TextInpaintingDataset( ## Creates a TextInpaintingDataset for the validation data, which will apply the specified masking strategy in a fixed manner for consistent evaluation.
            sequences=val_sequences,
            tokenizer=tokenizer,
            mask_type=mask_type,
            mask_ratio=mask_ratio,
            dynamic_masking=False,
        )

        train_loader = DataLoader( ## Creates a DataLoader for the training dataset, which will handle batching and shuffling of the data during training.
            train_data,
            batch_size=16,
            shuffle=True,
        )

        val_loader = DataLoader( ## Creates a DataLoader for the validation dataset, which will handle batching of the data during evaluation, without shuffling to maintain consistency.
            val_data,
            batch_size=16,
            shuffle=False,
        )

        model = DiffusionBert(T=T, conditioning_dropout=0.1).to(device) ## Initializes the DiffusionBert model with the specified number of diffusion steps (T) and conditioning dropout, and moves it to the appropriate device for training.

        diffusion_forward = DiscreteDiffusionForward(## Initializes the forward diffusion process with the specified number of steps (T) and the mask token ID from the tokenizer, and moves it to the appropriate device for training.
            T=T,
            mask_token_id=tokenizer.mask_token_id
        ).to(device)

        optimizer = torch.optim.AdamW( ## Sets up the AdamW optimizer with the model parameters and a learning rate of 3e-5 for training the diffusion model.
            model.parameters(),
            lr=3e-5,
        )
        print(train_loader.dataset[0].keys())
        for epoch in range(num_epochs):

            print(f"\nEpoch {epoch+1}/{num_epochs}")

            train_loss, train_acc = train_diffusion_epoch( ## Trains the diffusion model for one epoch by iterating over the training dataloader, sampling random timesteps, corrupting the target sequences according to the diffusion process, and computing the loss only on the masked positions to update the model parameters.
                model,
                train_loader,
                optimizer,
                diffusion_forward,
                tokenizer,
                device,
            )

            val_loss, val_acc = evaluate_diffusion( ## Evaluates the diffusion model on the validation set by following a similar process as training but without gradient updates, to compute the average loss and accuracy on the masked tokens.
                model,
                val_loader,
                diffusion_forward,
                tokenizer,
                device,
            )

            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Train Accuracy: {train_acc:.4f}")

            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_acc:.4f}")

            # SAVE BEST MODEL
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    model.state_dict(),
                    f"diffusion_{mask_type}_{mask_ratio}_T{T}_dropout_0.1.pt"
                )
                print("✅ Best model saved.")

        
    elif mode == "inference":

        def highlight_generated(original_ids, generated_ids, mask_positions, tokenizer): ## Highlights the generated tokens that were filled in for the masked positions by comparing the original token IDs, generated token IDs, and mask positions, and returns a string with the filled tokens highlighted in green.
            output_tokens = []

            for orig_id, gen_id, is_mask in zip( ##Iterates through the original token IDs, generated token IDs, and mask positions simultaneously to construct the output string with highlighted tokens.
                original_ids,
                generated_ids,
                mask_positions
            ):
                token = tokenizer.convert_ids_to_tokens(gen_id.item()) ## Converts the generated token ID back to its corresponding token string using the tokenizer.

                if is_mask:
                    # Green color for replaced tokens
                    token = f"\033[92m{token}\033[0m"

                output_tokens.append(token)

            return tokenizer.convert_tokens_to_string(output_tokens)
        
        print("\nRunning INFERENCE...\n")

        T = 12

        model = DiffusionBert( ## Initializes the DiffusionBert model with the specified number of diffusion steps (T) and conditioning dropout, and moves it to the appropriate device for inference.
            T=T,
            conditioning_dropout=0.1
        ).to(device)

        diffusion_forward = DiscreteDiffusionForward( ## Initializes the forward diffusion process with the specified number of steps (T) and the mask token ID from the tokenizer, and moves it to the appropriate device for inference.
            T=T,
            mask_token_id=tokenizer.mask_token_id
        ).to(device)

        model.load_state_dict( ## Loads the trained model weights from the specified file, mapping them to the appropriate device for inference.
            torch.load("diffusion_span_0.1_T12_dropout_0.1.pt", map_location=device)
        )

        model.eval()

        # Use validation dataset for testing
        val_data = TextInpaintingDataset( ## Creates a validation dataset for testing the inference performance of the diffusion model.
            sequences=val_sequences,
            tokenizer=tokenizer,
            mask_type="span",
            mask_ratio=0.20,
            dynamic_masking=False,
        )

        sample = val_data[0]
        ## Prepare input IDs and mask positions for the sample, moving them to the appropriate device for inference.
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        mask_positions = sample["mask_positions"].unsqueeze(0).to(device)

        mask_weights = torch.ones_like(input_ids, dtype=torch.float).to(device)
        mask_weights[mask_positions] = 1.0
        mask_weights[~mask_positions] = 0.2
        from inference.reverse_diffusion import reverse_diffusion_sample
        from analysis.span_id import get_span_ids

        span_ids = get_span_ids(mask_positions)


        generated,logits_steps, probs_steps = reverse_diffusion_sample( ## Runs the reverse diffusion sampling process using the trained model, forward diffusion process, tokenizer, input IDs, and mask positions to generate the inpainted token IDs for the masked positions.
            model,
            diffusion_forward,
            tokenizer,
            input_ids,
            mask_positions,
            T=T,
            temperature=0.8,
            top_k=20,
            device=device,
            guidance_fn=lambda logits, tokenizer, strength: span_guidance_with_penalty(
            logits,
            tokenizer,
            span_ids=span_ids[mask_positions],
            strength=strength
            ),
            guidance_strength=0.5,
            mask_weights=mask_weights
        )

        # from analysis.noise_analysis import compute_confidence, compute_entropy, aggregate_metrics, compute_confident_mistakes
        # from analysis.visualization import plot_metrics

        # confidence_steps = compute_confidence(probs_steps)
        # entropy_steps = compute_entropy(probs_steps)
        # mask_positions_cpu = mask_positions.cpu()

        # avg_conf, avg_ent = aggregate_metrics(confidence_steps, entropy_steps, mask_positions_cpu)

        # plot_metrics(avg_conf, avg_ent)

        # mistakes, total = compute_confident_mistakes(
        #     probs_steps,
        #     confidence_steps,
        #     sample["target_ids"].unsqueeze(0).cpu(), # ground truth
        #     mask_positions_cpu
        # )
        # from analysis.visualization import plot_confident_mistakes

        # plot_confident_mistakes(mistakes, total)

        # print("\nConfident Mistakes per Step:")
        # for i, (m, t) in enumerate(zip(mistakes, total)):
        #     rate = m / t if t > 0 else 0
        #     print(f"Step {i}: {m}/{t} (rate={rate:.4f})")

        # from analysis.noise_analysis import compute_confidence_histogram
        # from analysis.visualization import plot_confidence_histogram

        # step_counts = compute_confidence_histogram(
        # confidence_steps,
        # mask_positions_cpu
        # )

        # print("\nConfidence Histogram:", step_counts)

        # total_tokens = mask_positions_cpu.sum().item()

        # plot_confidence_histogram(step_counts, total_tokens)


        # from analysis.noise_analysis import compute_entropy_by_correctness
        # from analysis.visualization import plot_entropy_correct_vs_incorrect

        # entropy_correct, entropy_incorrect = compute_entropy_by_correctness(
        # probs_steps,
        # entropy_steps,
        # sample["target_ids"].unsqueeze(0).cpu(),
        # mask_positions_cpu
        # )

        # plot_entropy_correct_vs_incorrect(entropy_correct, entropy_incorrect)

        # from analysis.noise_analysis import prepare_entropy_heatmap
        # from analysis.visualization import plot_entropy_heatmaps

        # entropy_correct_mat, entropy_incorrect_mat = prepare_entropy_heatmap(
        # entropy_steps,
        # probs_steps,
        # sample["target_ids"].unsqueeze(0).cpu(),
        # mask_positions_cpu
        # )

        # plot_entropy_heatmaps(entropy_correct_mat, entropy_incorrect_mat)

        # from analysis.noise_analysis import compute_accuracy_per_step
        # from analysis.visualization import plot_accuracy_vs_step

        # accuracy_per_step = compute_accuracy_per_step(
        # probs_steps,
        # sample["target_ids"].unsqueeze(0).cpu(),
        # mask_positions_cpu
        # )

        # print("\nAccuracy per step:", accuracy_per_step)

        # plot_accuracy_vs_step(accuracy_per_step)

        from analysis.transition_analysis import extract_top_transitions, decode_tokens

        transitions = extract_top_transitions(probs_steps, mask_positions)

        for step in transitions:
            tokens = decode_tokens(step["tokens"], tokenizer)
            probs = step["probs"]

            print(f"\nStep {step['timestep']}:")
            for tok, p in zip(tokens, probs):
                print(f"{tok}: {p:.4f}")

        from analysis.transition_analysis import compute_stationary_distribution, print_top_stationary_tokens

        stationary = compute_stationary_distribution(probs_steps, mask_positions)
        print_top_stationary_tokens(stationary, tokenizer)


        from analysis.transition_analysis import compute_unigram_distribution, compare_stationary_unigram
        vocab_size = tokenizer.vocab_size
        unigram = compute_unigram_distribution(val_data, tokenizer, vocab_size)
        compare_stationary_unigram(stationary, unigram, tokenizer)
        similarity = F.cosine_similarity(stationary, unigram, dim=0)
        print(f"\nCosine Similarity: {similarity.item():.4f}")


        from analysis.graph_visualization import plot_transition_graph

        # choose a timestep (e.g., last step)

        step = transitions[11]

        tokens = decode_tokens(step["tokens"], tokenizer)
        probs = step["probs"]

        plot_transition_graph(tokens, probs, title=f"Step {step['timestep']} Transition")

        

        original_text = tokenizer.decode( ## Decodes the original target token IDs back into a human-readable string using the tokenizer, skipping any special tokens in the process.
            sample["target_ids"],
            skip_special_tokens=True
        )

        masked_text = tokenizer.decode( ## Decodes the masked input token IDs back into a human-readable string using the tokenizer, without skipping special tokens to show the masked positions clearly.
            sample["input_ids"],
            skip_special_tokens=False
        )

        highlighted_text = highlight_generated( ## Highlights the generated tokens that were filled in for the masked positions by comparing the original token IDs, generated token IDs, and mask positions, and returns a string with the filled tokens highlighted in green.
            sample["target_ids"],
            generated[0].cpu(),
            sample["mask_positions"],
            tokenizer
        )

        print("\n" + "="*80)
        print("ORIGINAL:\n")
        print(original_text)

        print("\n" + "="*80)
        print("MASKED:\n")
        print(masked_text)

        print("\n" + "="*80)
        print("GENERATED (Highlighted Filled Tokens):\n")
        print(highlighted_text)
        print("="*80)


    elif mode == "test":

        print("\nRunning TEST evaluation...\n")

        best_val_acc = 0.0
        T = 12
        mask_ratio = 0.1   
        batch_size = 16

        # Create test dataset
        test_data = TextInpaintingDataset(
            sequences=test_sequences,
            tokenizer=tokenizer,
            mask_type="span",
            mask_ratio=mask_ratio,
            dynamic_masking=False,  # IMPORTANT
        )

        test_loader = DataLoader( ## Creates a DataLoader for the test dataset, which will handle batching of the data during evaluation, without shuffling to maintain consistency.
            test_data,
            batch_size=batch_size,
            shuffle=False,
        )

        # Load trained model
        model = DiffusionBert(
            T=T,
            conditioning_dropout=0.1,
        ).to(device)

        model.load_state_dict( ## Loads the trained model weights from the specified file, mapping them to the appropriate device for evaluation.
            torch.load("diffusion_span_0.1_T12_dropout_0.1.pt", map_location=device)
        )

        diffusion_forward = DiscreteDiffusionForward( ## Initializes the forward diffusion process with the specified number of steps (T) and the mask token ID from the tokenizer, and moves it to the appropriate device for evaluation.
            T=T,
            mask_token_id=tokenizer.mask_token_id,
        ).to(device)

        # Run evaluation
        test_loss, test_acc = evaluate_diffusion(
            model,
            test_loader,
            diffusion_forward,
            tokenizer,
            device,
        )

        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        from analysis.confusion_matrix import compute_confusion_matrix, print_top_confusions
        from analysis.transition_matrix import compute_transition_matrix, print_transition_row
        
        vocab_size = tokenizer.vocab_size
        T_matrix = torch.zeros((vocab_size, vocab_size))
        all_confusion = None
    
        for batch_idx, batch in enumerate(test_loader):

        
            if batch_idx > 30:
                break

            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            mask_positions = batch["mask_positions"].to(device)

        generated, logits_steps,probs_steps = reverse_diffusion_sample(
            model=model,
            diffusion_forward=diffusion_forward,
            tokenizer=tokenizer,
            input_ids=input_ids,
            mask_positions=mask_positions,
            T=T,
            temperature=0.7,
            top_k=0,
            device=device
        )

        batch_T = compute_transition_matrix(
            probs_steps,
            target_ids.cpu(),
            mask_positions.cpu(),
            vocab_size
        )
        T_matrix += batch_T
        confusion = compute_confusion_matrix(
            generated.cpu(),
            target_ids.cpu(),
            mask_positions.cpu(),
            tokenizer
        )

        if all_confusion is None:
            all_confusion = confusion
        else:
            for k in confusion:
                for v in confusion[k]:
                    all_confusion[k][v] += confusion[k][v]
     
        ## normalise transition matrix
        row_sums = T_matrix.sum(dim=1, keepdim=True) + 1e-8
        T_matrix = T_matrix / row_sums

       
        ##print transition example
        token_id = tokenizer.convert_tokens_to_ids("a")
        print_transition_row(T_matrix, tokenizer, token_id)

        # print confusion matrix
        print_top_confusions(all_confusion)

        

        from analysis.pos_analysis import compute_pos_transitions, print_pos_transitions
        pos_transitions = compute_pos_transitions(all_confusion)
        print_pos_transitions(pos_transitions)

        # =========================
        # BLEU Evaluation
        # =========================

        # print("\nComputing BLEU Score...\n")

        # model.eval() ## Set the model to evaluation mode, which disables dropout and other training-specific behaviors, ensuring deterministic outputs during evaluation.

        # guidance_strengths = [0.5, 1.0, 1.5, 2.0]

        # for strength in guidance_strengths:
    
        #     print(f"\n--- Guidance Strength: {strength} ---")

        #     total_bleu = 0
        #     total_short = 0
        #     num_samples = 0

        #     for batch_idx, batch in enumerate(test_loader):

        #         if batch_idx > 30:
        #             break

        #         input_ids = batch["input_ids"].to(device)
        #         target_ids = batch["target_ids"].to(device)
        #         mask_positions = batch["mask_positions"].to(device)

        #         # create mask weights
        #         mask_weights = torch.ones_like(input_ids, dtype=torch.float).to(device)
        #         mask_weights[mask_positions] = 1.0
        #         mask_weights[~mask_positions] = 0.2

        #         generated, _, _ = reverse_diffusion_sample(
        #             model=model,
        #             diffusion_forward=diffusion_forward,
        #             tokenizer=tokenizer,
        #             input_ids=input_ids,
        #             mask_positions=mask_positions,
        #             T=T,
        #             temperature=0.7,
        #             top_k=0,
        #             device=device,
        #             guidance_fn=simple_guidance,
        #             guidance_strength=strength,
        #             mask_weights=mask_weights
        #         )

        #         for i in range(input_ids.size(0)):

        #             bleu = compute_masked_bleu(
        #                 target_ids[i].cpu(),
        #                 generated[i].cpu(),
        #                 mask_positions[i].cpu(),
        #                 tokenizer
        #             )

        #             short_pct = compute_short_token_percentage(
        #                 generated[i].cpu(),
        #                 mask_positions[i].cpu(),
        #                 tokenizer
        #             )

        #             total_bleu += bleu
        #             total_short += short_pct
        #             num_samples += 1

        #     avg_bleu = total_bleu / num_samples
        #     avg_short = total_short / num_samples

        #     print(f"BLEU: {avg_bleu:.4f}")
        #     print(f"% Short Tokens: {avg_short:.4f}")