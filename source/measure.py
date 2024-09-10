import numpy as np
import json
import sys
import re
import sacrebleu
from collections import Counter
from nltk.translate.meteor_score import meteor_score
import matplotlib.pyplot as plt


data = []
with open(sys.argv[1], "r", encoding="utf-8") as file:
    for line in file:
        data.append(json.loads(line))

def word_split(identifier):
    words = []
    if len(identifier) == 0:
        return []
    current_word = identifier[0]

    for char in identifier[1:]:
        if char.isdigit():
            # If char is a digit and the last char is not a digit, start a new word
            if not current_word or not current_word[-1].isdigit():
                words.append(current_word)
                current_word = char
            else:
                current_word += char
        elif char.isupper():
            # If char is uppercase and the last char is not uppercase, start a new word
            if not current_word or (not current_word[-1].isupper() or (current_word[-1].isupper() and current_word.islower())):
                words.append(current_word)
                current_word = char
            else:
                current_word += char
        elif char == "_":
            # If char is underscore and the previous char is not, start a new word
            if current_word != "":
                words.append(current_word)
                current_word = ""
        else:
            # Continue adding to the current word if it's lowercase
            current_word += char

    # Add the last word to the list
    words.append(current_word)
    return words

def group_strings_by_length(dicts):
    grouped_strings = {}
    
    for d in dicts:
        s = d["source"]
        length_step = ((len(s) // 300) + 1) * 300
        
        if length_step not in grouped_strings:
            grouped_strings[length_step] = []
        
        grouped_strings[length_step].append(d)
    return grouped_strings

def tiered_BLEU(data):
    scores = []
    data = group_strings_by_length(data)
    for key, array in data.items():
        references = []
        hypotheses = []
        for entry in array:
            references.append([entry["label"][0].replace("NEW_LINE", "\n").split("$")[-1]]) # .split("$")[2]
            prediction = entry["prediction"]
            match = re.search("([Ff]or )?[Ee]xample:?", prediction)
            print("ref", entry["label"])
            print("pred", prediction)
            if match:
                pass
            hypotheses.append(prediction)
        bleu_score = sacrebleu.corpus_bleu(hypotheses, references)
        scores.append((key, bleu_score.score))

    print(scores)
    x_values, y_values = zip(*scores)

    x_array = np.array(x_values)
    y_array = np.array(y_values)

    # Define bin size and range
    bin_size = 150 
    min_x = min(x_array)
    max_x = max(x_array)
    bins = range(int(min_x), int(max_x + bin_size), bin_size)

    # Average y-values within each bin
    averaged_data = {}
    for start in bins:
        end = start + bin_size
        mask = (x_array >= start) & (x_array < end)
        if np.any(mask):  # Check if there are any elements in this range
            averaged_y = np.mean(y_array[mask])
            averaged_data[(start + end) // 2] = averaged_y  # Store the average y for the midpoint of the range

    # Unpacking the averaged data
    averaged_x = list(averaged_data.keys())
    averaged_y = list(averaged_data.values())

    plt.figure(figsize=(10, 6))

    # Calculate the weighted average y for each unique x
    unique_xs = np.unique(x_array)
    weighted_ys = []
    average_ys = [np.mean(y_array[x_array == x]) for x in unique_xs]
    for x in unique_xs:
        indices = np.where(x_array == x)
        weights = len(indices[0])
        weighted_avg = np.sum(y_array[indices] * weights) / np.sum(weights)
        weighted_ys.append(weighted_avg)

    # Fit a polynomial of degree 3 (or other, as needed for your data) for smoothing
    coefficients = np.polyfit(unique_xs, average_ys, 3)
    polynomial = np.poly1d(coefficients)

    # Generate x values for the polynomial line (smooth curve)
    x_smooth = np.linspace(min(unique_xs), max(unique_xs), 500)
    y_smooth = polynomial(x_smooth)

    plt.plot(x_smooth, y_smooth, color='blue', label='Smoothed Weighted Average Line', linewidth=2)
    # Scatter plot of averaged data points
    plt.scatter(averaged_x, averaged_y, color='#008B8B', s=100, label='Averaged Data')

    # Adding labels and legend
    plt.ylabel('BLEU score')
    plt.xlabel('Character Count')
    plt.title('CodeImprove-docstring - Averaging BLEU scores within ±150 characters')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

    return scores


def measure_BLEU(data):
    references = []
    hypotheses = []

    for entry in data:
        references.append([entry["label"][0].replace("NEW_LINE", "\n").split("$")[-1]]) #.split("$")[2]
        prediction = entry["prediction"]
        match = re.search("([Ff]or )?[Ee]xample:?", prediction)
        print("ref", entry["label"])
        print("pred", prediction)
        if match:
            pass
        hypotheses.append(prediction)
    
    # Calculate corpus BLEU score
    bleu_score = sacrebleu.corpus_bleu(hypotheses, references)

    print(len(data))
    print("BLEU score:", bleu_score)
    print(list(zip(references, hypotheses))[:5])
    return bleu_score.score

def measure_TOP3EM(data):
    total = 0
    for entry in data:
        reference = entry["label"][0][0]
        hypo = entry["prediction"]

        matches = False
        for hyp in hypo:
            if reference == hyp:
                total += 1
                break

    return round(total / len(data) * 100)

def measure_TOP3UP(data):
    result = []
    for entry in data:
        reference = entry["label"][0]
        hypo = entry["prediction"]

        best = 0
        for hyp in hypo:
            sys.stdout.flush()
            ref = word_split(reference[0])
            h = word_split(hyp)
            print("ref", ref)
            print("h", h)

            if len(h) == 0:
                continue
            match = 0

            ref = dict(Counter(ref))
            total = len(ref)
            for word in h:
                if word in ref.keys():
                    ref[word] -= 1
                    if ref[word] == 0:
                        del ref[word]
                    match += 1
            score = min(match/total, match/len(h))
            if score > best:
                best = score
        result.append((best, len(entry["source"][0])))

    y_values, x_values = zip(*result)

    x_array = np.array(x_values)
    y_array = np.array(y_values)

    # Define bin size and range
    bin_size = 150 
    min_x = min(x_array)
    max_x = max(x_array)
    bins = range(int(min_x), int(max_x + bin_size), bin_size)

    # Average y-values within each bin
    averaged_data = {}
    for start in bins:
        end = start + bin_size
        mask = (x_array >= start) & (x_array < end)
        if np.any(mask):  # Check if there are any elements in this range
            averaged_y = np.mean(y_array[mask])
            averaged_data[(start + end) // 2] = averaged_y  # Store the average y for the midpoint of the range

    # Unpacking the averaged data
    averaged_x = list(averaged_data.keys())
    averaged_y = list(averaged_data.values())

    plt.figure(figsize=(10, 6))

    # Calculate the weighted average y for each unique x
    unique_xs = np.unique(x_array)
    weighted_ys = []
    average_ys = [np.mean(y_array[x_array == x]) for x in unique_xs]
    for x in unique_xs:
        indices = np.where(x_array == x)
        weights = len(indices[0])
        weighted_avg = np.sum(y_array[indices] * weights) / np.sum(weights)
        weighted_ys.append(weighted_avg)

    # Fit a polynomial of degree 3 (or other, as needed for your data) for smoothing
    coefficients = np.polyfit(unique_xs, average_ys, 3)
    polynomial = np.poly1d(coefficients)

    # Generate x values for the polynomial line (smooth curve)
    x_smooth = np.linspace(min(unique_xs), max(unique_xs), 500)
    y_smooth = polynomial(x_smooth)

    plt.plot(x_smooth, y_smooth, color='blue', label='Smoothed Weighted Average Line', linewidth=2)
    # Scatter plot of averaged data points
    plt.scatter(averaged_x, averaged_y, color='#008B8B', s=100, label='Averaged Data')

    # Adding labels and legend
    plt.ylabel('TOP3UP score')
    plt.xlabel('Character Count')
    plt.title('CodeImprove-rename - Averaging TOP3UP scores within ±150 characters')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

    print(result)
    return round(sum([r[0] for r in result]) / len(result) * 100, 2)


def measure_METEOR(data):
    for entry in data:
        reference = entry["label"][0]
        hypo = entry["prediction"]
        print(reference)
        print(hypo)

        for hyp in hypo:
            ref = reference.split("_")
            h = hyp.split("_") 
            print(ref, h)
            score = meteor_score([ref], h)
        return score

print(measure_BLEU(data))
