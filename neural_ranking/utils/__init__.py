import os
import dill

def load_rerank_data(path):
    packs = dill.load(open(os.path.join(path, "rerank.dill"), "rb"))
    return packs

def input_check():
    user_input = input("Continue? ")
    while user_input.strip().lower() != "y":
        user_input = input("Continue? ")