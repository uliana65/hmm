import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str,
                    help='path to data files (required)')
parser.add_argument('--test_data', type=str,
                    help='name of test data file in .tt format (required)')
parser.add_argument('--model_pred', type=str,
                    help="name of model's prediction file in .tt format (required)")

args = parser.parse_args()

# open the tagger output file and the gold standard file
file_gold = open(args.path + args.test_data)
file_model = open(args.path + args.model_pred)

precision_recall = {}

# read from files
n = 0
for line_gold, line_pred in zip(file_gold, file_model):
    line_gold = line_gold.rstrip()
    line_pred = line_pred.rstrip()
    n += 1

    if len(line_gold) != 0:

        word_gold, tag_gold = line_gold.split("\t")
        word_pred, tag_pred = line_pred.split("\t")

        # word forms should match in gold and system file
        if word_pred != word_gold:
            sys.exit("\nError in line " + str(n) + ": word mismatch!\n")

        if precision_recall.get(tag_pred) is None:
            precision_recall[tag_pred] = [0, 0, 0]
        if precision_recall.get(tag_gold) is None:
            precision_recall[tag_gold] = [0, 0, 0]

        precision_recall[tag_pred][1] += 1  # tag was assigned by system
        precision_recall[tag_gold][2] += 1  # tag was found in gold standard data

        # observe and count correct tags
        if tag_pred == tag_gold:
            precision_recall[tag_gold][0] += 1  # tag assignment was correct

# counts for overall accuracy
correct = 0
overall = 0

print(f"\nComparing gold file \"{args.test_data}\" and model's prediction file \"{args.model_pred}\"")
print("\nPrecision, recall, and F1 score:\n")

for tag, counts in precision_recall.items():

    # calculate precision, recall and F1 score, print them
    correct += counts[0]
    overall += counts[1]

    # precision, recall, and f1 for a subset of tags
    if 0 not in counts:
        precision = counts[0] / counts[1]
        recall = counts[0] / counts[2]
        f1_score = (2 * precision * recall) / (precision + recall)
        print("%5s %.4f %.4f %.4f" % (tag, precision, recall, f1_score))

print("\nAccuracy: %.4f\n" % (correct / overall))
