# Novel Model Context Protocol for Inclusive Education Apps

Implementation of multi-agent system for personalized educational interfaces for neurodivergent learners.

## Quick Start
```bash
# Clone repository
git clone https://github.com/Anonimous-Submission/anonymous-submission-xyz.git
cd anonymous-submission-xyz

# Create virtual environment (Python 3.11 required for textstat)
uv venv --python 3.11 .venv
source .venv/bin/activate

# Install requirements
uv pip install -r requirements.txt

# Build and run
mkdir build && cd build
cmake ..
make -j4
./test_orchestration -m ../models/Phi-3-mini-4k-instruct-q4.gguf
```

## Test Data
The `tests/` folder contains two JSON files used for evaluation:

- **`texts.json`**: 100 educational texts covering biology, physics, history, literature, and computer science. Each entry contains:
  ```json
  {
    "id": "text_0042",
    "content": "The water cycle describes how water evaporates..."
  }
  ```

- **`dialogs.json`**: 100 synthetic dialogue histories simulating user interactions. Each entry contains:
  ```json
  {
    "id": "dialogue_0001",
    "history": [
      {"role": "user", "content": "I have ADHD and find it hard to focus on long paragraphs..."}
    ]
  }
  ```

These files are used by `test_orchestration.cpp` to generate all 10,000 text-dialogue combinations (100×100) for evaluation, as described in the paper.

## Hardware Requirements
- 8GB+ VRAM (for 4-bit quantized models)
