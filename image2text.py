# Authors: ddholari-mjp5-rutdave

from PIL import Image
import math
import sys

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25
arrw = "-->"
TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "

def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    print(im.size)
    print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

# Code Starts
def load_input_data():
    content = []

    with open(train_txt_fname, 'r') as ip_file:
        for data in ip_file:
            inner_list = []
            for chunk in data.split():
                inner_list.append(chunk + ' ')
            content.append(inner_list)

    return content

def create_first_state():
    content = load_input_data()
    first_state = {}

    for data in content:
        for chunk in data:
            if(chunk[0] in TRAIN_LETTERS and chunk[0] not in first_state):
                first_state[chunk[0]] = 1
            elif(chunk[0] in TRAIN_LETTERS and chunk[0] in first_state):
                first_state[chunk[0]] += 1

    total_sum = 0
    for _, value in first_state.items():
        total_sum += value
    total_sum = float(total_sum)

    for key, value in first_state.items():
        first_state[key] = value / total_sum

    return first_state


def calculate_transition_probabilities(transition_letters):
    transition_count = {}

    for i in range(len(TRAIN_LETTERS)):
        count = 0

        for letter_pair in transition_letters:
            current_letter, _ = letter_pair.split('-->')
            
            if TRAIN_LETTERS[i] == current_letter:
                count += transition_letters[letter_pair]

        if count < 0:
            raise Exception("Count can't be negative")
        else:
            transition_count[TRAIN_LETTERS[i]] = count

    transition_probabilities = {}
    for letter_pair in transition_letters:
        current_letter, _ = letter_pair.split('-->')
        transition_probabilities[letter_pair] = transition_letters[letter_pair] / float(transition_count[current_letter])

    probability_total = {}
    total_probability = sum(transition_probabilities.values())
    for pair in transition_probabilities:
        probability_total[pair] = transition_probabilities[pair] / float(total_probability)

    return probability_total

def generate_transition_probabilities():
    input_data = load_input_data()
    transition_letters = {}

    for line in input_data:
        for words in line:
            for i in range(len(words) - 1):
                current_letter, next_letter = words[i], words[i + 1]
                letter_pair = f"{current_letter}{arrw}{next_letter}"

                if current_letter in TRAIN_LETTERS and next_letter in TRAIN_LETTERS:
                    if letter_pair not in transition_letters:
                        transition_letters[letter_pair] = 1
                    else:
                        transition_letters[letter_pair] += 1

    return calculate_transition_probabilities(transition_letters)

    
def process_test_letters(input_letters):
    count = [0, 0]
    for letter_set in input_letters:
        for row in letter_set:
            length = len(row)
            count[1] += length
            for index in range(length):
                if row[index] == '*':
                    count[0] += 1

    return count


def process_train_letters(input_letters):
    count = [0, 0]
    for letter_set in input_letters:
        for row in input_letters[letter_set]:
            length = len(row)
            count[1] += length
            for index in range(length):
                if row[index] == '*':
                    count[0] += 1

    return count

def calculate_emission_probabilities(test_letters, train_letters):
    emission_probabilities = {}

    test_counts = process_test_letters(test_letters)
    train_counts = process_train_letters(train_letters)

    for test_index in range(len(test_letters)):
        emission_probabilities[test_index] = {}

        for train_letter in train_letters:
            count = [0, 0]
            total_pixels = CHARACTER_WIDTH * CHARACTER_HEIGHT

            for i in range(len(test_letters[test_index])):
                for x in range(len(test_letters[test_index][i])):
                    test_char, train_char = test_letters[test_index][i][x], train_letters[train_letter][i][x]
                    
                    if test_char == train_char:
                        if train_char == '*':
                            count[0] += 1
                        elif train_char == ' ':
                            count[1] += 1

            ratio_test = test_counts[0] / test_counts[1]
            ratio_train = train_counts[0] / train_counts[1]

            if ratio_test >= ratio_train:
                emission_probabilities[test_index][train_letter] = 0.6 * (count[0] / total_pixels) + 0.4 * (count[1] / total_pixels)
            else:
                emission_probabilities[test_index][train_letter] = 0.9 * (count[0] / total_pixels) + 0.1 * (count[1] / total_pixels)

    return emission_probabilities


def calculate_simple_bayes(test_letters, train_letters):
    result_text = ''
    emission_probabilities = calculate_emission_probabilities(test_letters, train_letters)

    for value in emission_probabilities:
        max_probable_letter = max(emission_probabilities[value], key=emission_probabilities[value].get)
        result_text += ''.join(max_probable_letter)

    return result_text

def run_hmm_viterbi(test_letters, train_letters):
    initial_state = create_first_state()
    transition_probabilities = generate_transition_probabilities()
    emission_probabilities = calculate_emission_probabilities(test_letters, train_letters)

    output_sequence = ['X'] * len(test_letters)
    viterbi_matrix = []

    for _ in range(len(TRAIN_LETTERS)):
        viterbi_matrix.append([[0, ''] for _ in range(len(test_letters))])

    # Get initial state using initial state probabilities
    top_5_emissions = dict(sorted(emission_probabilities[0].items(), key=lambda x: x[1], reverse=True)[:5])
    for train_val in range(len(TRAIN_LETTERS)):
        if TRAIN_LETTERS[train_val] in initial_state and TRAIN_LETTERS[train_val] in top_5_emissions and top_5_emissions[TRAIN_LETTERS[train_val]] != 0:
            viterbi_matrix[train_val][0] = [-math.log(top_5_emissions[TRAIN_LETTERS[train_val]], 10), TRAIN_LETTERS[train_val]]

    max_value = 100000
    for test_val in range(1, len(test_letters)):
        top_5_emissions = dict(sorted(emission_probabilities[test_val].items(), key=lambda x: x[1], reverse=True)[:5])

        for val in top_5_emissions:
            subset_viterbi = {}
            for train_val in range(len(TRAIN_LETTERS)):
                transition_key = TRAIN_LETTERS[train_val] + arrw + val
                if transition_key in transition_probabilities and viterbi_matrix[train_val][test_val - 1][0] != 0:
                    subset_viterbi[val] = -40 * math.log(top_5_emissions[val], 10) - 0.009 * math.log(
                        transition_probabilities[transition_key], 10) - 0.009 * math.log(
                        viterbi_matrix[train_val][test_val - 1][0], 10)

            min_value = float('inf')
            final_letter = ''
            for key in subset_viterbi:
                if min_value > subset_viterbi[key]:
                    min_value = subset_viterbi[key]
                    final_letter = key
                viterbi_matrix[TRAIN_LETTERS.index(val)][test_val] = [subset_viterbi[final_letter], final_letter]

    for test_val in range(len(test_letters)):
        min_value = float('inf')
        for train_val in range(len(TRAIN_LETTERS)):
            if train_val < len(TRAIN_LETTERS) and viterbi_matrix[train_val][test_val][0] < min_value and viterbi_matrix[train_val][test_val][0] != 0:
                min_value = viterbi_matrix[train_val][test_val][0]
                output_sequence[test_val] = TRAIN_LETTERS[train_val]

    return ''.join(output_sequence)


print("Simple: ", calculate_simple_bayes(test_letters,train_letters))
print("HMM: ", run_hmm_viterbi(test_letters,train_letters))
