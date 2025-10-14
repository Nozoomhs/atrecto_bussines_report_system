This repository was creat to implement the solution for Attrecto's AI Developer challenge. Description of the solution can be found in [Blueprint.md](Blueprint.md)
Please excuse me for missing a t in the repo name :)
---
### Requirements and Setup
I assume an installed python, the tested environment is 3.12.11, conda virtual env.
The models consumed ~ 7GB of GPU memory, this is also currently a requirement. Hopefully it does not cause an issue.
Ollama:
Two ways to install it:
1. simply through: https://ollama.com/download
  After it is installed: ollama pull qwen2.5:7b-instruct
2. Through docker:
   docker run -d --name ollama -p 11434:11434 --gpus all -v ollama:/root/.ollama ollama/ollama
   
   docker exec -it ollama ollama pull qwen2.5:7b-instruct

In the environment `pip install -r requirements.txt`.
### How to run
1. Unzip data to folder named `data` from the root directory.
2. To run the entire pipeline(ingestion->parsing->agents->parsing->agent): `python run_entire_pipeline.py` - Runtime is ~ 1 minute on Nvidia Geforce RTX 4080.
