import numpy as np
import nltk.corpus.reader as crc
from collections import Counter
import sys
import argparse
from collections import defaultdict
from progress.bar import Bar


def viterbi_alg(tagset, initial, transitions, emissions, input):
    """
    Optimsation of search for the most probable sequence based on Viterbi algorithm.
    :param tagset: list of possible tags
    :param initial: initial probabilities of tags
    :param transitions: transition probabilities of tag-to-tag
    :param emissions: emission probabilities emission probabilities of words (how likely is it that the given tag emits this word?)
    :param input: sentence as list of tokens
    :return: most likely sequence of hidden states as list of tags
    """
    back_pointers = dict()
    best_paths_probs = dict()

    # initialize time steps (ts) and best_paths_probs (bp)
    for ts in range(1, len(input)+1):
        best_paths_probs[ts] = np.array([])
        back_pointers[ts] = list()

    # GET BEST PATH
    # at time step (ts) x
    for ts in range(1, len(input)+1):
        # for each state (tag)
        for tag in tagset:
            all_probs_at_current = np.array([])
            if ts == 1:
                # we are at the initial state
                try:
                    # count probs for initial states if this word was seen with this tag
                    # in initial state in training
                    probs = emissions[input[ts-1]][tagset.index(tag)] * initial[tag]
                except KeyError:
                    # otherwise take prior prob of the tag
                    probs = 1 * initial[tag]
                # ts: ([tag1prob, tag2prob, ...])
                all_probs_at_current = np.append(all_probs_at_current, probs)
                # take max and store them
                best_paths_probs[ts] = np.append(best_paths_probs[ts], max(all_probs_at_current))
                # store a bp for ts: (current_tag, initial, prob)
                back_pointers[ts].append((tag, "initial", max(all_probs_at_current)))

            else:
                # we are at the transition state
                try:
                    # count probs for transitions from all states to current state
                    # if this word was seen with this tag in training
                    probs = emissions[input[ts-1]][tagset.index(tag)] * transitions[tag] * best_paths_probs[ts-1]
                except KeyError:
                    # otherwise only take prob of the tag-to-tag transition
                    probs = 1 * transitions[tag] * best_paths_probs[ts-1]
                # ts: ([tag1prob, tag2prob, ...])
                all_probs_at_current = np.append(all_probs_at_current, probs)
                # take max and store them
                best_paths_probs[ts] = np.append(best_paths_probs[ts], max(all_probs_at_current))
                # store a bp for ts: (current_tag, tag_before, prob)
                back_pointers[ts].append((tag, tagset[np.where(all_probs_at_current == max(all_probs_at_current))[0][0]], max(all_probs_at_current)))


    # get state sequence
    sequence = sequence_decode(back_pointers)

    if len(input) != len(sequence):
        print(f'input: {input}\nseq: {sequence}')
    return sequence


def sequence_decode(bps):
    """
    This function finds the most likely sequence from back pointers
    :param bps: back pointers
    :return: sequence of hidden states
    """
    sequence = list()

    #pick the best sequence
    max_x = ""
    max_prob = 0
    for output in bps[len(bps)]:
        if max_prob < output[2]:
            max_x = output[0]
            max_prob = output[2]
    sequence.append(max_x)

    #reconstruct the path
    prev_x = ""
    for i in reversed(range(1, len(bps)+1)):
        for output in bps[i]:
            if output[0] == sequence[-1]:
                prev_x = output[1]
        sequence.append(prev_x)

    #remove initial state
    if "initial" in sequence:
        sequence.remove("initial")
    else:
        sequence = sequence[:-1]

    #reverse the sequence
    sequence.reverse()

    return sequence


def hmm_train(corpus):
    """
    Learns initial, transition, emission probabilities from training data
    :param corpus: training/eval dataset
    :return: set of possible tags (aka hidden states), probability dictionaries
    """
    tagset = set()
    initial = defaultdict(int)
    transitions = dict()
    emissions = dict()
    # use to get transition probs
    tag_seq = ["<SB>"]
    # use to get emission probs
    bigrams_counts = defaultdict(int)
    with Bar('Learning initial probabilities...', max=len(corpus)) as bar:
        # learn initial freqs, get pos tags and word sequence
        for sent in corpus:
            initial[sent[0][1]] += 1
            for word in sent:
                tagset.add(word[1])
                tag_seq.append(word[1])
                bigrams_counts[word] += 1
            tag_seq.append("<SB>")
            bar.next()
        tagset = list(tagset)

        # normalize initial probs
        for tag in initial.keys():
            initial[tag] = initial[tag] / len(corpus)

    words = set([word[0] for sent in corpus for word in sent])
    with Bar('Learning emission probabilities...', max=len(words)) as bar:
        # learn emission probs
        for word in words:
            emissions[word] = [0.0 for tag in tagset]
            for tag in tagset:
                try:
                    ind = tagset.index(tag)
                    prob = bigrams_counts[(word, tag)] / tag_seq.count(tag)
                    emissions[word][ind] = emissions[word][ind] + prob
                except KeyError:
                    continue
            bar.next()

    with Bar('Learning transition probabilities...', max=len(tagset)) as bar:
        # learn transition probs
        for tag in tagset:
            transitions[tag] = np.array([0.0 for tag in tagset])
        bigrams_tags = [(tag_seq[i], tag_seq[i+1]) for i in range(len(tag_seq)-1)]
        bigrams_tags_count = Counter(bigrams_tags)
        for tag_to in tagset:
            for tag_from in tagset:
                ind = tagset.index(tag_from)
                tag_from_count = sum([bigrams_tags_count[bigr] for bigr in bigrams_tags_count.keys() if bigr[0] == tag_from and bigr[1] != "<SB>"])
                prob = bigrams_tags_count[(tag_from, tag_to)] / tag_from_count
                transitions[tag_to][ind] = transitions[tag_to][ind] + prob
            bar.next()

    return tagset, initial, transitions, emissions


def reader(path, file):
    """
    This method handles the dataset.
    :param path: full path to the folder with the datasets
    :param file: train/test/eval file name, .tt format
    :return: read-in corpus, ConllCorpusReader object
    """
    if "train" or "eval" in file:
        corpus = crc.ConllCorpusReader(path, file, ["words", "pos"])
    else:
        # testing
        corpus = crc.ConllCorpusReader(path, file, ["words"])
    return corpus


def pos_tagger(path, tagset, initial, transitions, emissions, corpus):
    """
    Taggs hidden states of sequences based  on learned probabilities and writes the result in a file.
    :param tagset: set of possible tags
    :param initial: initial probabilities of tags
    :param transitions: transition probabilities of tag-to-tag
    :param emissions: emission probabilities of words (how likely is it that the given tag emits this word?)
    :param corpus: read-in test corpus
    :return: None
    """
    with Bar('Predicting test data sequences...', max=len(corpus.sents())) as bar:
        for sent in corpus.sents():
            pos_seq = viterbi_alg(tagset, initial, transitions, emissions, sent)
            with open(path+"test_output.tt", "a", encoding="utf-8") as f:
                for i in range(len(sent)):
                    if i+1 == len(sent):
                        # end of set
                        tagged_pair = f"{sent[i]}\t{pos_seq[i]}\n\n"
                        f.write(tagged_pair)
                    else:
                        tagged_pair = f"{sent[i]}\t{pos_seq[i]}\n"
                        f.write(tagged_pair)
            bar.next()


if __name__ == "__main__":
    # TRAINING ON CORPUS
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str,
                        help='path to data files (required)')
    parser.add_argument('--train_data', type=str,
                        help='name of training data file in .tt format (required)')
    parser.add_argument('--test_data', type=str,
                        help='name of test data file in .tt format (required)')

    args = parser.parse_args()

    corpus = reader(args.path, args.train_data)
    tagset, initial, transitions, emissions = hmm_train(corpus.tagged_sents()[:])
    print(f"Finished training...")

    test_set = reader(args.path, args.test_data)
    pos_tagger(args.path, tagset, initial, transitions, emissions, test_set)


