
CUDA_VISIBLE_DEVICES="0,1" coverage run -a --data-file=/workspace/.coverage --source=/workspace/ -m pytest tests/unit_tests -m "not pleasefixme" --with_downloads
