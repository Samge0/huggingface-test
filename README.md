## huggingface的模型快速测试

[huggingface](https://huggingface.co/)

- env for windows - 50.48
```shell
conda create --name graphrag_test python==3.10.13 -y

pip install G:/whl_dir/windows/cuda-py310/torch-2.1.0+cu118-cp310-cp310-win_amd64.whl && `
pip install G:/whl_dir/windows/cuda-py310/torchvision-0.16.0+cu118-cp310-cp310-win_amd64.whl && `
pip install G:/whl_dir/windows/cuda-py310/torchaudio-2.1.0+cu118-cp310-cp310-win_amd64.whl && `
pip install G:/whl_dir/windows/cuda-py310/xformers-0.0.22.post4+cu118-cp310-cp310-win_amd64.whl 

pip install transformers
```