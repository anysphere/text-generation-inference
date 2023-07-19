include Makefile-flash-att
include Makefile-flash-att-v2
include Makefile-vllm

unit-tests:
	/home/azureuser/miniconda3/envs/dev/bin/python -m pytest -s -vv -m "not private" tests

gen-server:
	# Compile protos
	/home/azureuser/miniconda3/envs/dev/bin/pip install grpcio-tools==1.51.1 mypy-protobuf==3.4.0 'types-protobuf>=3.20.4' --no-cache-dir
	mkdir text_generation_server/pb || true
	/home/azureuser/miniconda3/envs/dev/bin/python -m grpc_tools.protoc -I../proto --python_out=text_generation_server/pb \
		--grpc_python_out=text_generation_server/pb --mypy_out=text_generation_server/pb ../proto/generate.proto
	find text_generation_server/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;
	touch text_generation_server/pb/__init__.py

install-torch:
	# Install specific version of torch
	/home/azureuser/miniconda3/envs/dev/bin/pip install torch --extra-index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

install: gen-server install-torch
	/home/azureuser/miniconda3/envs/dev/bin/pip install pip --upgrade
	/home/azureuser/miniconda3/envs/dev/bin/pip install -r requirements.txt
	/home/azureuser/miniconda3/envs/dev/bin/pip install -e ".[bnb, accelerate]"

run-dev:
	SAFETENSORS_FAST_GPU=1 /home/azureuser/miniconda3/envs/dev/bin/python -m torch.distributed.run --nproc_per_node=2 text_generation_server/cli.py serve bigscience/bloom-560m --sharded

export-requirements:
	/home/azureuser/miniconda3/envs/dev/bin/poetry export -o requirements.txt -E bnb --without-hashes
