import re
import string

text = "The day was good. uhdfgfhjfghfv hbdfuygfgfjf 12345 hgf"

words = re.split(r"\s", text)

lim = 5
sym = ">>> " # space after it

#letters = list(string.ascii_lowercase)
letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
thing = [",", "."]

print("|".join(re.escape(x) for x in thing))

for let in letters:
    print(let)
    print(bool(re.search("|".join(re.escape(x) for x in thing), let)))


def phrase_split(user_text, lim):
    phrase = ""
    counter = 0

    words = re.split(r"\s", user_text)

    letters_pattern = "|".join(letters)
    thing_pattern = "|".join(re.escape(x) for x in thing)

    for word in range(len(words)):
        word1 = words[word]
        word2 = words[word + 1] if word + 1 < len(words) else ""

        for i in range(len(word1)):
            counter += 1

            letter = word1[i]
            letter2 = word1[i + 1] if i + 1 < len(word1) else ""

            if counter == 1:
                new = "\n" + sym + letter

            elif (counter % lim == 0 and not re.search(letters_pattern, letter2)
                and not re.search(thing_pattern, word2) and not re.search(letters_pattern, word2)):
                new = "\n" + sym + letter

            elif (counter % lim == 0 and not re.search(letters_pattern, letter2)):
                new = "\n" + sym + letter

            elif (counter % lim == 0 and re.search(letters_pattern, letter) and re.search(letters_pattern, letter2)
                and not re.search(thing_pattern, letter)):
                new = "-\n" + sym + letter

            elif (counter % lim == 0 and (letter2 == "" or re.search(thing_pattern, letter))):
                new = "\n" + sym + letter

            elif (counter % lim == 0 and re.search(letters_pattern, letter) and re.search(letters_pattern, letter2) and re.search(thing_pattern, letter)):
                new = "\n" + sym + letter

            elif (counter % lim == 0 and re.search(thing_pattern, letter)):
                new = "\n" + sym + letter

            else:
                new = letter

            phrase += new

        phrase += " "

    return phrase[:-1]


print(phrase_split(text, lim))

