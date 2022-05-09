docker build -t 'dockermutualinf:latest' \
	--no-cache \
	--build-arg TRANSFORMERS_CACHE=/partisanbrain/transformers_cache \
	.
