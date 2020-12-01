import math
import pickle
import torch
import torch.nn as nn
from torchtext import data
from torchtext.data.utils import get_tokenizer

from TransformerModel import TransformerModel

if __name__ == '__main__':
    # Create Field objects
    CONTEXT = data.Field(tokenize=get_tokenizer('basic_english'),
                         lower=True)
    RESPONSE = data.Field(tokenize=get_tokenizer('basic_english'),
                         lower=True,
                         is_target=True)

    # Create column tuples
    fields = [
        ('context', CONTEXT),
        ('response', RESPONSE)
    ]

    # Load the dataset
    train_ds, valid_ds, test_ds = data.TabularDataset.splits(
        path = 'data/preprocessed',
        train = 'train.tsv',
        validation = 'validation.tsv',
        test = 'test.tsv',
        format = 'tsv',
        fields = fields,
        skip_header = False
    )
    print('Loaded data')

    # They need the same vocab, but this is an inefficient way to do that
    CONTEXT.build_vocab(train_ds.context)
    RESPONSE.build_vocab(train_ds.context)
    print('Built vocab')

    device = torch.device('cpu')

    # Pad and numericalize data
    train_contexts = [example.context for example in train_ds.examples]
    # It's probably learning mostly <pad> with a large size, so try smaller
    max_train_context_size = 20 # max([len(context) for context in train_contexts])
    train_contexts = CONTEXT.numericalize([context[:min(len(context), max_train_context_size)] + ['<pad>']*(max_train_context_size - len(context)) for context in train_contexts]).to(device)
    train_responses = [example.response for example in train_ds.examples]
    # responses and contexts need to be same size
    max_train_response_size = max_train_context_size # max([len(response) for response in train_responses])
    train_responses = CONTEXT.numericalize([response[:min(len(response), max_train_response_size)] + ['<pad>']*(max_train_response_size - len(response)) for response in train_responses]).to(device)

    test_contexts = [example.context for example in test_ds.examples]
    max_test_context_size = max([len(context) for context in test_contexts])
    test_contexts = CONTEXT.numericalize([context[:min(len(context), max_train_context_size)] + ['<pad>']*(max_train_context_size - len(context)) for context in test_contexts]).to(device)
    test_responses = [example.response for example in test_ds.examples]
    max_test_response_size = max([len(response) for response in test_responses])
    test_responses = CONTEXT.numericalize([response[:min(len(response), max_train_response_size)] + ['<pad>']*(max_train_response_size - len(response)) for response in test_responses]).to(device)

    valid_contexts = [example.context for example in valid_ds.examples]
    max_valid_context_size = max([len(context) for context in valid_contexts])
    valid_contexts = CONTEXT.numericalize([context[:min(len(context), max_train_context_size)] + ['<pad>']*(max_train_context_size - len(context)) for context in valid_contexts]).to(device)
    valid_responses = [example.response for example in valid_ds.examples]
    max_valid_response_size = max([len(response) for response in valid_responses])
    valid_responses = CONTEXT.numericalize([response[:min(len(response), max_train_response_size)] + ['<pad>']*(max_train_response_size - len(response)) for response in valid_responses]).to(device)

    print('Generated contexts and responses')

#    test_contexts = CONTEXT.numericalize([test_ds.examples[0].context])
#    test_responses = RESPONSE.numericalize([test_ds.examples[0].response])
#    test_contexts = test_contexts.view(-1, 1).t().contiguous().to(device)
#    test_responses = test_responses.view(-1, 1).t().contiguous().to(device)
#
#    valid_contexts = CONTEXT.numericalize([valid_ds.examples[0].context])
#    valid_responses = RESPONSE.numericalize([valid_ds.examples[0].response])
#    valid_contexts = valid_contexts.view(-1, 1).t().contiguous().to(device)
#    valid_responses = valid_responses.view(-1, 1).t().contiguous().to(device)
    

    # Model hyperparameters
    ntokens = len(CONTEXT.vocab.stoi)
    insize = 200
    outsize = 200
    nhid = 200
    nlayers = 2
    nhead = 2
    dropout = 0.1
    model = TransformerModel(ntokens, insize, outsize, nhead, nhid, nlayers, dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 5.0
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    print('Created Torch objects')

    bptt = 15
    def get_batch(contexts, targets, i):
        context = contexts[:, i].view(1, -1)
        target = targets[:, i].view(1, -1)
        retctx = torch.zeros(bptt, context.size(1), dtype=context.dtype)
        retres = torch.zeros(bptt, target.size(1), dtype=target.dtype)
        for j in range(bptt):
            retctx[j, :] = context
            retres[j, :] = target
        return retctx, retres

#    def one_hot(vals, vocab):
#        """
#        Turn vals (tensor of size 1 x context length) into one-hot array
#        (tensor of size context length x vocab size)
#        """
#        retval = torch.zeros(vals.size(1), len(vocab.stoi))
#        for i, val in enumerate(vals[0]):
#            vocab_index = val.item()
#            retval[i, vocab_index] = 1
#        # print('retval:', retval)
#        return retval

    import time
    def train():
        model.train() # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        ntokens = len(CONTEXT.vocab.stoi)
        # print('Started training')
        for batch, i in enumerate(range(0, train_contexts.size(0)-1), bptt):
            # print('batch:', batch, 'i:', i)
            data, targets = get_batch(train_contexts, train_responses, i)
            # print('got batch')
            optimizer.zero_grad()
            # print('zeroed gradient')
            target_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
            # print('generated new mask, if needed')
            output = model(data, targets, target_mask)
            # print('data:', data)
            # print(data.size())
            # print('target_mask:', target_mask)
            # print(target_mask.size())
            # print(output)
            # print(output.size())
            # print('generated output')
            # print('targets:', targets)
            # print('targets.size():', targets.size())
            loss = criterion(output.view(-1, ntokens), targets.view(-1))
            # print('loss:', loss)
            loss.backward()
            # print('loss backward')
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            # print('clip gradient')
            optimizer.step()
            # print('step')

            total_loss += loss.item()
            log_interval = 50
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                        epoch, batch, len(train_contexts) // bptt, scheduler.get_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

    def evaluate(eval_model, data_source, data_targets):
        eval_model.eval() # Turn on the evaluation mode
        total_loss = 0.
        ntokens = len(CONTEXT.vocab.stoi)
        with torch.no_grad():
            # print('data_source.size(0):', data_source.size(0))
            for i in range(0, data_source.size(0)-1):# - 1, bptt):
                # print('i:', i)
                data, targets = get_batch(data_source, data_targets, i)
                target_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)
                output = eval_model(data, targets, target_mask)
                output_flat = output.view(-1, ntokens)
                # print('output_flat:', output_flat)
                # print('targets:', targets)
                total_loss += len(data) * criterion(output_flat, targets.view(-1)).item()
                # print('total_loss:', total_loss)
        return total_loss / (len(data_source) - 1)

    best_val_loss = float("inf")
    epochs = 10 # The number of epochs
    best_model = None
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train()
        # print(valid_contexts)
        # print('\n\n')
        # print(valid_responses)
        val_loss = evaluate(model, valid_contexts, valid_responses)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
    
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model
    
        scheduler.step()

    test_loss = evaluate(best_model, test_contexts, test_responses)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    with open('model.pickle', 'wb') as handle:
        pickle.dump(best_model, handle)
    with open('CONTEXT.pickle', 'wb') as handle:
        pickle.dump(CONTEXT, handle)

