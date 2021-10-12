FROM huggingface/transformers-pytorch-gpu

RUN pip3 install --upgrade pip
# RUN apt-get update
RUN pip3 install ipython
RUN pip3 install openai
RUN pip3 install --upgrade transformers
RUN mkdir -p /partisanbrain/transformers_cache
RUN export TRANSFORMERS_CACHE=/partisanbrain/transformers_cache
WORKDIR "/partisanbrain"
RUN cd /partisanbrain
# RUN export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

ENTRYPOINT ["/bin/bash"]
