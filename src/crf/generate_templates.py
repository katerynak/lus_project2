import itertools


def generate_template(w_token=3, w_pos=5, bigram=False, w_token_b=3, w_pos_b=5):
    filename = "./templates/wtok_{}_wpos_{}".format(w_token, w_pos)
    if bigram:
        filename += "_bigram_wtok_{}_wpos_{}".format(w_token_b, w_pos_b)

    with open(filename, 'w') as f:
        f.write("# Unigram\n")
        i=0
        for j in range(-int(w_token/2), int(w_token/2)+1):
            f.write("U{}:%x[{},0]\n".format(i,j))
            i+=1
        f.write("")
        for j in range(-int(w_token/2), int(w_token/2)):
            f.write("U{}:%x[{},0]/%x[{},0]\n".format(i,j,j+1))
            i+=1
        f.write("")
        for j in range(-int(w_token/2), int(w_token/2)-1):
            f.write("U{}:%x[{},0]/%x[{},0]/%x[{},0]\n".format(i,j,j+1,j+2))
            i+=1
        f.write("")

        for j in range(-int(w_pos/2), int(w_pos/2)+1):
            f.write("U{}:%x[{},1]\n".format(i,j))
            i+=1
        f.write("")
        for j in range(-int(w_pos/2), int(w_pos/2)):
            f.write("U{}:%x[{},1]/%x[{},1]\n".format(i,j,j+1))
            i+=1
        f.write("")
        for j in range(-int(w_pos/2), int(w_pos/2)-1):
            f.write("U{}:%x[{},1]/%x[{},1]/%x[{},1]\n".format(i,j,j+1,j+2))
            i+=1
        f.write("")

        if bigram:
            i=0
            f.write("\n# Bigram\n")
            for j in range(-int(w_token_b / 2), int(w_token_b / 2) + 1):
                f.write("B{}:%x[{},0]\n".format(i, j))
                i += 1
            f.write("")
            for j in range(-int(w_token_b / 2), int(w_token_b / 2)):
                f.write("B{}:%x[{},0]/%x[{},0]\n".format(i, j, j + 1))
                i += 1
            f.write("")
            for j in range(-int(w_token_b / 2), int(w_token_b / 2) - 1):
                f.write("B{}:%x[{},0]/%x[{},0]/%x[{},0]\n".format(i, j, j + 1, j + 2))
                i += 1
            f.write("")

            for j in range(-int(w_pos_b / 2), int(w_pos_b / 2) + 1):
                f.write("B{}:%x[{},1]\n".format(i, j))
                i += 1
            f.write("")
            for j in range(-int(w_pos_b / 2), int(w_pos_b / 2)):
                f.write("B{}:%x[{},1]/%x[{},1]\n".format(i, j, j + 1))
                i += 1
            f.write("")
            for j in range(-int(w_pos_b / 2), int(w_pos_b / 2) - 1):
                f.write("B{}:%x[{},1]/%x[{},1]/%x[{},1]\n".format(i, j, j + 1, j + 2))
                i += 1
            f.write("")


if __name__== "__main__":
    w_tokens = [7]
    w_poss = [3, 5, 7]
    w_tokens_b = [1, 3]
    w_poss_b = [1, 3]

    params_bigram = [w_tokens, w_poss, [True], w_tokens_b, w_poss_b]
    params_configs_bigram = list(itertools.product(*params_bigram))

    for param in params_configs_bigram:
        if param[0] >= param[1]:
            if param[3] >= param[4]:
                generate_template(*param)
