import re

string = "The Day123 was go0od.!?"

# extract lowercase only
lwr_wrds = re.sub("([^a-z])+", " ", string)

# make letters only
ltrs = re.sub("([^a-zA-Z])+", " ", string)

# make numbers only
nbrs = re.sub("([^0-9])+", " ", string)

# make punctuation only
punc = re.sub("[[:punct:]]", " ", string)

# make numbers and letters only
nbrs_ltrs = re.sub("([^0-9a-zA-Z])+", " ", string)

# make lowercase
lwr = string.lower()

# make uppercase
upr = string.upper()

# split after fullstop
split_string = str.split(string, ".")

# capitalise only the first word
#cap = str.title(string)
cap = '. '.join(list(map(lambda x: x.strip().capitalize(), string.split('.'))))