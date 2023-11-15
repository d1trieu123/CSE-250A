words = {}

file_path = "hw1_word_counts_05.txt"

# this puts all the txt file into a dictionary
with open(file_path, "r") as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) == 2:
            word, number = parts[0], parts[1]
        words[word] = int(number)


# (A)
# sorts the dictionary by frequency of word
sorted_word_number_reverse = dict(
    sorted(words.items(), key=lambda item: item[1], reverse=False)
)
sorted_word_number = dict(sorted(words.items(), key=lambda item: item[1], reverse=True))

# top 15 and bottom 14 frequency words
top_15_pairs = list(sorted_word_number.items())[:15]
bottom_14_pairs = list(sorted_word_number_reverse.items())[:14]





# calculates the probability of selecting the word from the entire list of words based on values P(W=w) = COUNT(W) / Σ(w')COUNT(w')
word_probability = {}
for word, count in words.items():
    total_words = sum(words.values())
    word_probability[word] = count / total_words


# (B)
def get_next_guess(known, correct, incorrect):
    possible_guesses = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") - (correct | incorrect)
    next_guess = ""
    next_prob = 0.0
    possible_words = []

    for word, prob in word_probability.items():  # loops through each word in the corpus checking the compatability of each word given the known, correct, and incorrect guesses
        compatible = True
        for spot, char in enumerate(word):  # loops through each letter in the word
            if char in correct:  # if the letter is in the correct guesses but the letter is in the wrong place, it cannot be the word
                if known[spot] != char:
                    compatible = False
                    break
            elif char in incorrect:# if the letter has already been guessed and it is incorecct, it cannot be the word
                compatible = False
                break
            elif known[spot] != "-" and known[spot] != char:  # if the letter has been revealed, and it is not the letter in the corpus word, it cannot be the word
                compatible = False
                break
        if compatible: # creates a an array containing all the possible words given the evidence
            possible_words.append(word)
    


 
    sum_of_possible_words = 0 #denom of the first term
    for word in possible_words:
        sum_of_possible_words += word_probability[word]
    
    for letter in possible_guesses:
        letter_prob = 0.0
        for word in possible_words:
            second_term = 0
            if letter in word:
                second_term = 1
            else:
                second_term = 0
            first_term = word_probability[word]/sum_of_possible_words
            letter_prob += first_term * second_term
        if letter_prob > next_prob:
            next_guess=letter
            next_prob= letter_prob
    print(next_guess, next_prob)



