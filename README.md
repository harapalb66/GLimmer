# GLIMMER: GLoss-based Image Multiword Meaning Expression Ranker

Official implementation of GLIMMER for the SemEval-2026 ADMIRE shared task.

## Overview

GLIMMER is a hybrid retrieval-reasoning system that ranks images by how well they express the idiomatic or literal meaning of multiword expressions (MWEs) across 15 languages. The system combines:

- **Gloss-based semantic anchoring** using GPT-5.1 to generate definitional glosses
- **Dual embedding scoring** with multilingual text encoders and CLIP vision-language models
- **LLM semantic verification** for nuanced figurative language understanding

## System Architecture

GLIMMER operates in three stages:

1. **Gloss Generation**: Generate definitional glosses for MWEs using GPT-5.1
2. **Embedding-based Retrieval**: Compute similarity between glosses and image captions using:
   - Text path: SentenceTransformers (paraphrase-multilingual-mpnet-base-v2)
   - Vision path: CLIP (ViT-B-32)
3. **LLM Semantic Verification**: Re-rank candidates using GPT-5.1 semantic scoring

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/glimmer.git
cd glimmer

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for CLIP)
- OpenAI API key

## Setup

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### Data Format

The system expects data in the following structure:

```
data_root/
  <language>/
    <compound>/
      <image_id>.png
  subtask_a_<split>.tsv
```

The TSV file should contain columns: `compound`, `sentence`, `sentence_type`, `image1_name`, `image1_caption`, etc.

### Running GLIMMER

```python
from system import GLIMMERSystem

# Initialize the system
glimmer = GLIMMERSystem(
    openai_model="gpt-5.1",
    weight_text_vs_clip=0.6,
    weight_embeddings_vs_llm=0.4
)

# Process a language
glimmer.process_language(
    language_name="Greek",
    lang_code="EL",
    data_root="./test",
    tsv_path="./test/admire_file/submission_Greek.tsv",
    output_path="./submissions/submission_EL.tsv"
)
```

### Command Line

Run the system on test data:

```bash
python system.py
```

This will process all 15 languages and generate submission files in the `submissions/` directory.

## Configuration

Key hyperparameters:

- `openai_model`: LLM model for gloss generation and scoring (default: "gpt-5.1")
- `weight_text_vs_clip`: Text vs CLIP weighting (default: 0.6)
- `weight_embeddings_vs_llm`: Embedding vs LLM weighting (default: 0.4)
- `text_model_name`: Sentence encoder (default: "paraphrase-multilingual-mpnet-base-v2")
- `clip_model_name`: CLIP model (default: "ViT-B-32")

## Performance

GLIMMER achieved competitive performance on SemEval-2026 Task 1:

- Overall accuracy: 51.2%
- Best performance: Portuguese-Brazil (66.7%), Russian (62.9%), Slovenian (58.8%)
- Zero-shot transfer across 15 languages without language-specific tuning


## License

MIT License - see LICENSE file for details

## Contact

For questions or issues, please open an issue on GitHub 
