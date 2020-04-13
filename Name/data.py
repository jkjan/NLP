from __future__ import unicode_literals, print_function, division
import pickle
from io import open
import glob
import os
import unicodedata
import string

all_letters = string.ascii_letters + ".,;'"
n_letters = len(all_letters)

def find_files(path): return glob.glob(path)

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def read_lines(file_name):
    lines = open(file_name, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

def get_name_dict():
    try:
        category_lines = pickle.load(open("category_lines", "rb"))
        all_categories = pickle.load(open("all_categories", "rb"))

    except FileNotFoundError:
        print("pickle data does not exist.")

        category_lines = {}
        all_categories = []

        for file_name in find_files('../data/English/names/data/names/*.txt'):
            category = os.path.splitext(os.path.basename(file_name))[0]
            all_categories.append(category)
            lines = read_lines(file_name)
            category_lines[category] = lines

        with open("category_lines", "wb") as save:
            pickle.dump(category_lines, save)

        with open("all_categories", "wb") as save:
            pickle.dump(all_categories, save)

    return all_categories, category_lines