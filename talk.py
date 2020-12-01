# Interact with chatbot


import math
import pickle

import torch


if __name__ == '__main__':
    with open('model.pickle', 'rb') as handle:
        best_model = pickle.load(handle)
    with open('CONTEXT.pickle', 'rb') as handle:
        CONTEXT = pickle.load(handle)

    ########
    # Start talking
    ########
    in_sentence = input('Say something: ')
    in_context = CONTEXT.numericalize([in_sentence.lower().split(' ')])
    response = torch.zeros((15,1), dtype=in_context.dtype)
    # Say 15 words
    for wordnum in range(15):
        #print(in_context)
        #print(in_context.size())
        #print('response:', response)
        #print(response.size())
        output = best_model(in_context, response, None)
#        tmp_ctx = best_model.input_encoder(in_context) * math.sqrt(best_model.ninp)
#        print('input_encoder:', tmp_ctx)
#        tmp_ctx = best_model.pos_input_encoder(tmp_ctx)
#        print('pos_encoder:', tmp_ctx)
#        tmp_ctx = best_model.transformer_encoder(tmp_ctx)
#        print('trans_encoder:', tmp_ctx)
#        tmp_res = best_model.output_encoder(response) * math.sqrt(best_model.noutp)
#        print('output_encoder:', tmp_res)
#        tmp_res = best_model.pos_output_encoder(tmp_res)
#        print('pos_encoder:', tmp_res)
#        tmp_out = best_model.transformer_decoder(tmp_res, tmp_ctx, None)
#        print('trans_decoder:', tmp_out)
#        tmp_out = best_model.linear_decoder(tmp_out)
#        print('linear_decoder:', tmp_out)
#        biggest_val = float('-inf')
#        biggest_index = -1
#        for i, val in enumerate(tmp_out[0, 0, :]):
#            if val > biggest_val:
#                biggest_val = val
#                biggest_index = i
#        print('biggest_index:', biggest_index)
#        print(CONTEXT.vocab.itos[biggest_index])

        next_word_val = float('-inf')
        next_word_index = -1
        for i, val in enumerate(output[-1, 0, :]):
            if val > next_word_val:
                next_word_val = val
                next_word_index = i
        response[wordnum, 0] = next_word_index
#        print(response)

    # Say the response
    response_words = []
    for i in range(response.size(0)):
        response_words.append(CONTEXT.vocab.itos[response[i,0]])
    print(' '.join(response_words))
