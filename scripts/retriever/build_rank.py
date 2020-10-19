#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import code
import prettytable
import logging
import os
import io
from drqa import retriever


def process(query, k=1):
    doc_names, doc_scores = ranker.closest_docs(query, k)
    table = prettytable.PrettyTable(
        ['Rank', 'Doc Id', 'Doc Score']
    )
    for i in range(len(doc_names)):
        table.add_row([i + 1, doc_names[i], '%.5g' % doc_scores[i]])
    print(table)


def rank(args):
    logger.info('Initializing ranker...')
    ranker = retriever.get_class('tfidf')(tfidf_path=args.model)

    basename = os.path.splitext(os.path.basename(args.data_path))[0]
    dump_path = os.path.join(args.out_dir, f'{basename}-{args.k}.rank')
    logger.info(f'Dumping rank jsons to {dump_path}')

    with io.open(args.data_path) as json_file:
        for idx, line in enumerate(json_file):
            if idx % 1000 == 0:
                logger.info(f'\t{idx} finished...')

            input_json = json.loads(line.strip('\n'))
            doc_id, doc = input_json['id'], input_json['text']

            doc_names, doc_scores = ranker.closest_docs(query=doc, k=args.k)

            dump_json = {
                'doc_id': doc_id,
                'rank_ids': doc_names,
                'rank_scores': doc_scores,
            }
            json_str = json.dumps(dump_json, ensure_ascii=False)
            with open(dump_path, 'a') as f:
                f.write(json_str+'\n')


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--data_path', type=str, help='/path/to/data')
    parser.add_argument('--out_dir', type=str, help='/path/to/out_dir')

    args = parser.parse_args()
    rank(args)
    # python scripts/retriever/build_rank.py --k 50 --model /home/s1617290/DrQA/cnndm/train-tfidf-ngram=2-hash=16777216-tokenizer=corenlp.npz --data_path /home/s1617290/DrQA/cnndm/train.json --out_dir /home/s1617290/DrQA/cnndm/ 