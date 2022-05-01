import argparse
import pickle
import numpy as np

from metrics import global_measure, local_measure

def global_local(args):
    id2word = pickle.load(open(args.dic_id2word, "rb"))
    vec_t1 = np.load(args.path_models[0])
    vec_t2 = np.load(args.path_models[1])

    target_words = ["actually",
                    "must",
                    "promise",
                    "gay",
                    "cell"]
    
    word2id = {}
    for word_id, word in id2word.items():
        word2id[word] = word_id

    for target_word in target_words:
        target_id = word2id[target_word]
        vec_i_t1 = vec_t1[target_id]
        vec_i_t2 = vec_t2[target_id]
        g_dis = global_measure(vec_i_t1, vec_i_t2)
        l_dis = local_measure(vec_t1, vec_t2, len(word2id), target_id, topn=25)
        print(f"{target_word}\tglobal: {g_dis}, local: {l_dis}")
    

def cli_main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--dic_id2word", help="path of dict[id]: word (pickle file)")
    parser.add_argument("-m", "--path_models", nargs="*", help="path of wordvectors (pickle file)")
    args = parser.parse_args()
    global_local(args)

if __name__ == "__main__":
    cli_main()
