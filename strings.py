import re

string = "The Day123 was go0od.!?"

# extract lowercase only
lwr_wrds = re.sub("([^a-z])+", " ", string)

# make letters only
ltrs = re.sub("([^a-zA-Z])+", " ", string)

# make numbers only
nbrs = re.sub("([^0-9])+", " ", string)

# make punctuation only
#punc = re.compile("[[:punct:]]")
punc = re.compile("[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]")
punc = punc.findall(string)

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

################################################################################

string1 = "The.Day//Was1111111Good for the 10 of the month being 10/03/2024 a Sunday"

# extract everything after the //
ext = re.sub("\\W\\W+", " ", string1)

# extract date only
date = re.compile("\\d+/\\d+/\\d+")
date = date.findall(string1)

# get numbers not joined to any words
std_nm = re.compile("\\s\\d+\\s")
std_nm = std_nm.findall(string1)