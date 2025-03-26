import itertools
import json
import random

from Wrapper_Z3 import Wrapper_Z3

with open("../Models/json/Oryx2.json", "r") as file:
    application = json.load(file)

with open("../Data/json/offers_20.json", "r") as file:
    offers_do = json.load(file)

# get all keys from the original object
keys = list(offers_do.keys())

# For SecureWebContainer, Oryx2
# generate all possible combinations of r=36 keys out of 40 approx which leads to approx 90 k
# itertools.combinations generates the combinations in lex order of they key which in our case is the Id of the offer.
# Hence, there might be offers with very similar hardware chosen, especially the first ones generated.
# For Oryx2, we stopped at 6925 samples which took more than 2 days to generate.
# For Wordpress3
# generate all possible combinations of r=37 keys out of 40 approx which leads to approx 9880
# for 40 choose 30/10 takes a lot of time to generate milions of numbers
# used 20 choose 7
combos = list(itertools.combinations(keys, 7))

# So, we shuffle the keys in order to obtain a good variety of offers in a generated offer
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
    with open("../Models/json/Oryx2.json", "r") as file:
        application = json.load(file)
    # TODO It seems that the LowerBound between 2 components does not work well when creating the output. E.g. {'type': 'LowerBound', 'compsIdList': [1, 2], 'bound': 3} but it should be 'compsIdList': [2, 3],
    result = wrapper.solve(application, offer)
    if result:
        index = index + 1
        print("idx", index)
        with open(f"../Datasets/DsOryx2_20_7/{application['application']}_{index}.json", "w") as outfile:
            # Write the JSON data to the file
            json.dump(result, outfile, indent=4)