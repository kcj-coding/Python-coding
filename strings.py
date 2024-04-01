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

# can also be written
punc = re.findall(r"[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]", string)

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

### extract everything after the // ###
ext = re.sub("\\W\\W+", " ", string1)
# or
ext = re.compile("(?://{1,})([^*]+)")
#(?://{1,}) # find 1 or more instances of //
#([^*]+) # extract everything after this
ext = ext.findall(string1)

# extract date only
date = re.compile("\\d+/\\d+/\\d+")
date = date.findall(string1)
# or
date_2 = re.compile("(?://{1,})*(\d+/\d+/\d+)")
#(?://{1,}) # find 1 or more instances of //
#(\d+/\d+/\d+) # extract numbers in the format d/d/d after this
date_2 = date_2.findall(string1)

# get numbers not joined to any words
std_nm = re.compile("\\s\\d+\\s")
std_nm = std_nm.findall(string1)
# or
std_nm2 = re.findall(r"(?://{1,})*\s(\d+)\s",string1)
#(?://{1,}) # find 1 or more instances of //
#\s(\d+)\s # extract numbers in the format whitespace number whitespace

################################################################################

string2 = "The.Weather.is.nice.today./12abcdef"

# remove fullstops and ignore everything after the /
simple = re.findall("([^.]+)(?:\.)", string2)
# ([^.]+) # extract everything and ignore any . characters
# (?:\.) # do not capture anything after the /

# extract the number after the /
num_simple = re.compile("\\/([0-9]+)")
num_simple = num_simple.findall(string2)

# extract 2 letter words
two_ltr = re.findall(r"(\b\w{2,2}\b)", string2)
# (\b\w{2,2}\b) # find only whole words at word boundary \b and only show those of length 2 characters

################################################################################

string3 = "thezzzzzzzzz day \\\\\ is be goood cc days"

# extract 2 letter words after the \\\\\
#ext1 = re.findall(r"(?:\\{1,})*\s(\w{2,2})(?=\s|$)",string3)
# (?:\\{1,})* #
# \s(\w{2,2}) #
# (?=\s|$) # 
ext1 = re.findall(r"\b\w{2,2}\b", string3)

# only return whole words with the letter a in them
a = re.findall(r"(?=[a-z]*a)[a-z]+", string3)
# (?<![a-z]) # negative lookbehind # not needed
# (?=[a-z]*a) # positive lookahead to any words with a in them
# [a-z]+ # capture any words matching this condition

# extract only non duplicated letter words
non_dupe_ltr = re.findall(r"\b(?!\w*?(\w)\1)(\w+)", string3)

# find instances of repeated letters in the string
rptd_ltrs = re.findall(r"(\w+)\1", string3)


