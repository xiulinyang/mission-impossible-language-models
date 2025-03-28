# hop_surprisal.py
# Author: Julie Kallini

# For importing utils
import sys
sys.path.append("..")

import os
import torch
import pandas as pd
import tqdm
import argparse
from numpy.random import default_rng
from transformers import GPT2LMHeadModel
from gpt2_no_positional_encoding_model import GPT2NoPositionalEncodingLMHeadModel
from itertools import zip_longest
from glob import glob
from utils import CHECKPOINT_READ_PATH, PERTURBATIONS, PAREN_MODELS, \
    BABYLM_DATA_PATH, gpt2_hop_tokenizer, \
    marker_sg_token, marker_pl_token, compute_surprisals


MAX_TRAINING_STEPS = 900
CHECKPOINTS = list(range(100, MAX_TRAINING_STEPS+1, 100))

def compute_anchor_surprisal(model, input_ids, reverse=False):
    """
    Computes the surprisal (negative log2 probability) for a target token relative to the <anchor> token.

    In non-reverse mode (reverse=False):
      - Assumes <anchor> is placed before the original first token.
      - Computes the surprisal for the token immediately following the anchor.

    In reverse mode (reverse=True):
      - Assumes <anchor> is placed immediately after the original first token.
      - Computes the surprisal for the <anchor> token itself, using the logits from the token preceding it.

    Parameters:
      model: The language model.
      input_ids: A tensor of token ids (shape: [batch_size, sequence_length]).
      anchor_token_id: The integer id representing the '<anchor>' token.
      reverse: Boolean flag indicating which configuration to use.

    Returns:
      A list of surprisal values (negative log2 probabilities) for the target token in each sequence.
      If the expected token (or the preceding token in reverse mode) is missing, returns None for that sequence.
    """
    anchor_token_id = marker_sg_token
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]
        batch_surprisals = []

        for i in range(input_ids.shape[0]):
            sequence = input_ids[i]
            # Locate the first occurrence of the <anchor> token in the sequence
            anchor_positions = (sequence == anchor_token_id).nonzero(as_tuple=False)
            if anchor_positions.numel() == 0:
                # No anchor token found in this sequence
                batch_surprisals.append(None)
                continue

            anchor_index = anchor_positions[0].item()

            if not reverse:
                # Non-reverse: <anchor> comes before the original first token.
                target_index = anchor_index + 1
                if target_index >= sequence.size(0):
                    # No token follows the anchor.
                    batch_surprisals.append(None)
                    continue
                target_token_id = sequence[target_index]
                # Logits at anchor_index predict the token at anchor_index+1.
                token_logits = logits[i, anchor_index]
            else:
                # Reverse: <anchor> is placed right after the original first token.
                # Here, the target is the <anchor> token at anchor_index,
                # but its prediction comes from the logits at anchor_index-1.
                if anchor_index == 0:
                    # Cannot compute surprisal if <anchor> is the first token.
                    batch_surprisals.append(None)
                    continue
                target_index = anchor_index  # The <anchor> token itself.
                target_token_id = sequence[target_index]
                token_logits = logits[i, anchor_index - 1]

            # Convert logits to a probability distribution and then to log probabilities (base 2).
            log_probs = torch.log2(torch.nn.functional.softmax(token_logits, dim=-1))
            surprisal = -log_probs[target_token_id].item()
            batch_surprisals.append(surprisal)

        return batch_surprisals

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Get marker token surprisals for hop verb perturbations',
        description='Marker token surprisals')
    parser.add_argument('perturbation_type',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=PERTURBATIONS.keys(),
                        help='Perturbation function used to transform BabyLM dataset')
    parser.add_argument('train_set',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=["100M", "10M"],
                        help='BabyLM train set')
    parser.add_argument('random_seed', type=int, help="Random seed")
    parser.add_argument('paren_model',
                        default='all',
                        const='all',
                        nargs='?',
                        choices=list(PAREN_MODELS.keys()) + ["randinit"],
                        help='Parenthesis model')
    parser.add_argument('-np', '--no_pos_encodings', action='store_true',
                        help="Train GPT-2 with no positional encodings")

    # Get args
    args = parser.parse_args()
    no_pos_encodings_underscore = "_no_positional_encodings" if args.no_pos_encodings else ""


    # Get path to model
    model = f"babylm_{args.perturbation_type}_{args.train_set}_{args.paren_model}{no_pos_encodings_underscore}_seed{args.random_seed}"
    model_path = f"{CHECKPOINT_READ_PATH}/babylm_{args.perturbation_type}_{args.train_set}_{args.paren_model}{no_pos_encodings_underscore}/{model}/runs/{model}/checkpoint-"

    # Get perturbed test files
    test_files = sorted(glob(BABYLM_DATA_PATH +
        "/babylm_data_perturbed/babylm_{}/babylm_test_affected/*".format(args.perturbation_type)))

    EOS_TOKEN = gpt2_hop_tokenizer.eos_token_id
    FILE_SAMPLE_SIZE = 1000
    MAX_SEQ_LEN = 1024
    rng = default_rng(args.random_seed)

    marker_token_sequences = []
    nomarker_token_sequences = []
    target_indices = []

    # Iterate over data files to get surprisal data
    print("Sampling BabyLM affected test files to extract surprisals...")
    for test_file in test_files:
        print(test_file)

        # Get tokens from test file (+ eos token), and subsample
        f = open(test_file, 'r')
        file_token_sequences = [
            [int(s) for s in l.split()] + [EOS_TOKEN] for l in f.readlines()]
        file_token_sequences = [
            toks for toks in file_token_sequences if len(toks) < MAX_SEQ_LEN]
        sample_indices = rng.choice(
            list(range(len(file_token_sequences))), FILE_SAMPLE_SIZE, replace=False)
        file_token_sequences = [file_token_sequences[i]
                                for i in sample_indices]

        file_target_indices = []
        file_nomarker_token_sequences = []
        for tokens in file_token_sequences:
            # Find index of first marker token for surprisal target
            target_index = None
            for idx in range(len(tokens)):
                if tokens[idx] in (marker_sg_token, marker_pl_token):
                    target_index = idx
                    break
            assert (target_index is not None)

            # Make a version of tokens with marker removed at surprisal target
            nomarker_tokens = tokens.copy()
            nomarker_tokens.pop(target_index)
            assert (tokens[:target_index] == nomarker_tokens[:target_index])
            assert (tokens[target_index] in (marker_sg_token, marker_pl_token))
            assert (tokens[target_index+1] == nomarker_tokens[target_index])

            file_target_indices.append(target_index)
            file_nomarker_token_sequences.append(nomarker_tokens)

        marker_token_sequences.extend(file_token_sequences)
        nomarker_token_sequences.extend(file_nomarker_token_sequences)
        target_indices.extend(file_target_indices)

    # For logging/debugging, include decoded sentence
    marker_sents = [gpt2_hop_tokenizer.decode(
        toks) for toks in marker_token_sequences]
    nomarker_sents = [gpt2_hop_tokenizer.decode(
        toks) for toks in nomarker_token_sequences]

    surprisal_df = pd.DataFrame({
        "Sentences with Marker": marker_sents,
        "Sentences without Marker": nomarker_sents,
    })

    BATCH_SIZE = 32
    device = "cuda"
    for ckpt in CHECKPOINTS:
        print(f"Checkpoint: {ckpt}")

        # Load model
        if args.no_pos_encodings:
            model = GPT2NoPositionalEncodingLMHeadModel.from_pretrained(
                model_path + str(ckpt)).to(device)
        else:
            model = GPT2LMHeadModel.from_pretrained(
                model_path + str(ckpt)).to(device)

        # Init lists for tracking correct/wrong surprisals for each example
        marker_token_surprisals = []
        nomarker_token_surprisals = []

        # Iterate over data in batches
        for i in tqdm.tqdm(range(0, len(marker_token_sequences), BATCH_SIZE)):

            # Extract data for batch and pad the sequences
            marker_batch = marker_token_sequences[i:i+BATCH_SIZE]
            correct_batch_padded = zip(
                *zip_longest(*marker_batch, fillvalue=gpt2_hop_tokenizer.eos_token_id))
            marker_batch_tensors = torch.tensor(
                list(correct_batch_padded)).to(device)

            # Do the same for wrong batch
            nomarker_batch = nomarker_token_sequences[i:i+BATCH_SIZE]
            nomarker_batch_padded = zip(
                *zip_longest(*nomarker_batch, fillvalue=gpt2_hop_tokenizer.eos_token_id))
            nomarker_batch_tensors = torch.tensor(
                list(nomarker_batch_padded)).to(device)

            # Get target indices in a batch
            targets_batch = target_indices[i:i+BATCH_SIZE]

            # Compute marker/nomarker surprisals for batches
            # marker_surprisal_sequences = compute_surprisals(
            #     model, marker_batch_tensors)
            # nomarker_surprisal_sequences = compute_surprisals(
            #     model, nomarker_batch_tensors)

            # UPDATED Compute marker/nomarker surprisals for batches
            if 'acw' in args.perturbation_type:
                marker_surprisal_sequences = compute_anchor_surprisal(
                    model, marker_batch_tensors, reverse=True)
                nomarker_surprisal_sequences = compute_surprisals(
                    model, nomarker_batch_tensors)
            else:
                marker_surprisal_sequences = compute_anchor_surprisal(
                    model, marker_batch_tensors)
                nomarker_surprisal_sequences = compute_surprisals(
                    model, nomarker_batch_tensors)

            # Extract surprisals for target token
            for marker_seq, nomarker_seq, idx in \
                    zip(marker_surprisal_sequences, nomarker_surprisal_sequences, targets_batch):
                marker_token_surprisals.append(marker_seq)
                nomarker_token_surprisals.append(nomarker_seq[idx])

        # Add surprisals to df
        ckpt_df = pd.DataFrame(
            list(zip(marker_token_surprisals, nomarker_token_surprisals)),
            columns=[f'Marker Token Surprisals (ckpt {ckpt})',
                     f'No Marker Token Surprisals (ckpt {ckpt})']
        )
        surprisal_df = pd.concat((surprisal_df, ckpt_df), axis=1)

    # Write results to CSV
    directory = f"hop_surprisal_results/{args.perturbation_type}_{args.train_set}{no_pos_encodings_underscore}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    file = directory + f"/{args.paren_model}_seed{args.random_seed}.csv"
    print(f"Writing results to CSV: {file}")
    surprisal_df.to_csv(file)
