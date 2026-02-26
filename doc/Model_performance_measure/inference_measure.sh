""" Tested on "timm==1.0.14" 

    Model Name List:
    {spanetv2_s18_pure, spanetv2_s18_hybrid
    convformer_s18, caformer_s18,
    }
"""
MODEL_NAME=spanetv2_s18_pure
BATCH_SIZE=128


python benchmark.py --model $MODEL_NAME --bench inference --batch-size $BATCH_SIZE\
 --img-size 224 --num-warm-iter 10 --num-bench-iter 1000 --fuser te --device cuda:0
