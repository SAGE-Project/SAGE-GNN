import itertools
import json
import random

from Wrapper_Z3 import Wrapper_Z3

with open("../Models/json/SecureWebContainer.json", "r") as file:
    application = json.load(file)

with open("../Data/json/offers_40.json", "r") as file:
    offers_do = json.load(file)

# get all keys from the original object
keys = list(offers_do.keys())

# For SecureWebContainer
# generate all possible combinations of r=36 keys out of 40 approx which leads to approx 90 k
# itertools.combinations generates the combinations in lex order of they key which in our case is the Id of the offer.
# Hence, there might be offers with very similar hardware chosen, especially the first ones generated.
combos = list(itertools.combinations(keys, 36))

# So, we shuffle the keys in order to obtain a good variety of offers in an offer
random.shuffle(combos)

offers_comb = []
# loop through each combination and create a new object
for combo in combos:
    new_obj = {}
    for key in combo:
        new_obj[key] = offers_do[key]
    offers_comb.append(new_obj)

index = 0
for offer in offers_comb:
    print("offer ", offer)
    wrapper = Wrapper_Z3()
    with open("../Models/json/SecureWebContainer.json", "r") as file:
        application = json.load(file)
    result = wrapper.solve(application, offer)
    if result:
        index = index + 1
        print("idx", index)
        with open(f"../Datasets/DsSecureWebContainer/{application['application']}_{index}.json", "w") as outfile:
            # Write the JSON data to the file
            json.dump(result, outfile, indent=4)