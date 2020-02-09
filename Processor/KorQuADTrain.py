import json

corpus_fname = "../raw/KorQuAD_v1.0_train.json"
output_fname = "../processed/processed_korquad_train.txt"

with open(corpus_fname) as f1, open(output_fname, "w", encoding="UTF-8") as f2:
    dataset_json = json.load(f1)
    dataset = dataset_json['data']
    for article in dataset:
        w_lines = []
        for paragraph in article['paragraphs']:
            w_lines.append(paragraph['context'])
            for qa in paragraph['qas']:
                q_text = qa['question']
                for a in qa['answers']:
                    a_text = a['text']
                    w_lines.append(q_text + " " + a_text)
        for line in w_lines:
            f2.writelines(line + "\n")