# English_Spanish
Machine Translation experiments on the English Spanish language pair.

Experiment 1
Parameters:
Dimension of representation space: 300
Number of layers in the LSTM: 2
Batch size: 128
Number of steps per checkpoint: 100

Spanish vocabulary (types) size: 883431
English vocabulary (types) size: 883799

Number of training bisegments: 15337051
Number of English training words (tokens): 245177685
Number of Spanish training words (tokens): 277355099

Number of tuning bisegments: 2502
Number of English tuning words (tokens): 44819
Number of Spanish tuning words (tokens): 51202


Number of test bisegments: 2511
Number of English test words (tokens): 44630
Number of Spanish test words (tokens): 51351


BLEU = 29.00, 60.1/35.0/22.5/15.3 (BP=0.995, ratio=0.995, hyp_len=44402, ref_len=44630)

