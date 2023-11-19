import itertools
import json
from Wrapper_Z3 import Wrapper_Z3

with open("../Models/json/SecureWebContainer.json", "r") as file:
    application = json.load(file)

with open("../Data/json/offers_20.json", "r") as file:
    offers_do = json.load(file)

# get all keys from the original object
keys = list(offers_do.keys())

# generate all possible combinations of 10 keys out of 20 approx 180 k -> for Secure Web Container generate 20 k
# Better r = 15 which leads to approx 15 k
combos = itertools.combinations(keys, 10)

offers_comb = []
# loop through each combination and create a new object
for combo in combos:
    new_obj = {}
    for key in combo:
        new_obj[key] = offers_do[key]
    offers_comb.append(new_obj)

index = 0
for offer in offers_comb:
    wrapper = Wrapper_Z3()
    with open("../Models/json/SecureWebContainer.json", "r") as file:
        application = json.load(file)
    result = wrapper.solve(application, offer)
    if result:
        index = index + 1
        with open(f"../Datasets/DsSecureWeb/{application['application']}_{index}.json", "w") as outfile:
            # Write the JSON data to the file
            json.dump(result, outfile, indent=4)