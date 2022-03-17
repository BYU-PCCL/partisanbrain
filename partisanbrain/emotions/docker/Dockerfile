FROM huggingface/transformers-pytorch-gpu

RUN pip install --upgrade pip
RUN pip install ipython
RUN pip install --upgrade transformers
RUN pip install --upgrade torch
RUN pip install sklearn
RUN pip install openai
RUN pip install pandas
RUN pip install matplotlib
RUN pip install lifelines
RUN pip install cmocean
RUN mkdir -p /partisanbrain/transformers_cache
ENV TRANSFORMERS_CACHE=/partisanbrain/transformers_cache
WORKDIR "/partisanbrain"
RUN cd /partisanbrain

ENTRYPOINT ["/bin/bash"]
