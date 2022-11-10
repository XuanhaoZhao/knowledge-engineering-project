import os
from nltk.corpus.reader.conll import ConllCorpusReader
from tokenizer_tools.tagset.NER.BILUO import BILUOEncoderDecoder

from corpus import get_corpus_reader
import utils
import re

current_dir = os.path.dirname(os.path.abspath(__file__))

conllu_dir = os.path.join(current_dir, 'data/split_data')
utils.create_if_dir_not_exists(conllu_dir)

token_dir = os.path.join(current_dir, 'data/split_char_softmax')
utils.create_if_dir_not_exists(token_dir)

encoder_cache = {}


def main(delimit="  "):
    for root, dirs, files in os.walk(conllu_dir):
        for file_ in files:

            # ignore hidden file
            if os.path.basename(file_).startswith('.'):
                continue

            corpus_reader = get_corpus_reader(
                columntypes=(
                    ConllCorpusReader.IGNORE,
                    ConllCorpusReader.WORDS,
                    ConllCorpusReader.POS,
                    ConllCorpusReader.NE,
                    ConllCorpusReader.IGNORE
                ),
                root=root,
                fileids=[file_]
            )

            output_file = os.path.splitext(file_)[0] + '.char.bmes'

            with open(os.path.join(token_dir, output_file), 'w') as fd:
                for word_tag_pair_list in corpus_reader.tagged_sents():
                    for word, tag in word_tag_pair_list:
                        if tag not in encoder_cache:
                            encoder_cache[tag] = BILUOEncoderDecoder(tag)

                        encoder = encoder_cache[tag]
                        coding = encoder.encode(word)

                        word_coding_pair_list = zip(word, coding)

                        for word_coding_pair in word_coding_pair_list:
                            fd.write(delimit.join(word_coding_pair) + "\n")
                    fd.write('\n')

def updateFile(file,old_str,new_str):
    with open(file, "r", encoding="utf-8") as f1,open("%s.bak" % file, "w", encoding="utf-8") as f2:
        for line in f1:
            f2.write(re.sub(old_str,new_str,line))
    os.remove(file)
    os.rename("%s.bak" % file, file)
    
if __name__ == "__main__":
    main()
    updateFile(r"./data/split_char_softmax/dev.char.bmes", "L-LOC", "I-LOC")
    updateFile(r"./data/split_char_softmax/test.char.bmes", "L-LOC", "I-LOC")
    updateFile(r"./data/split_char_softmax/train.char.bmes", "L-LOC", "I-LOC")
    updateFile(r"./data/split_char_softmax/dev.char.bmes", "L-PER", "I-PER")
    updateFile(r"./data/split_char_softmax/test.char.bmes", "L-PER", "I-PER")
    updateFile(r"./data/split_char_softmax/train.char.bmes", "L-PER", "I-PER")
    updateFile(r"./data/split_char_softmax/dev.char.bmes", "L-ORG", "I-ORG")
    updateFile(r"./data/split_char_softmax/test.char.bmes", "L-ORG", "I-ORG")
    updateFile(r"./data/split_char_softmax/train.char.bmes", "L-ORG", "I-ORG")
