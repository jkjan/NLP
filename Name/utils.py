import torch
import random
from data import all_letters, n_letters

def letter_to_index(letter):
    return all_letters.find(letter)

def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor

def category_from_output(all_categories, output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()

    return all_categories[category_i], category_i


def random_choice(l):
    return l[random.randint(0, len(l)-1)]

def random_training_example(all_categories, category_lines):
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = line_to_tensor(line)

    return category, line, category_tensor, line_tensor


