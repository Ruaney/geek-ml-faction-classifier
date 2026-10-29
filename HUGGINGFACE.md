# Publish to Hugging Face

## 1. Train the model

```bash
pip install -r requirements-hf.txt
python train_hf.py --epochs 8
```

The trained model will be saved at `artifacts/hf/model`.

## 2. Authenticate

Create a Hugging Face account, generate a **write** access token in its settings, then run:

```bash
hf auth login
```

The token is entered locally and is not added to this repository.

## 3. Upload the model

```bash
python hf_upload.py --repo-id Ruaney/geek-genre-text-classifier
```

Use your own Hugging Face username if it differs from your GitHub username.

## 4. Create a Space

Create a new public **Gradio** Space in the Hugging Face interface. Copy the contents of the `space/` folder into that Space. Set the `MODEL_ID` environment variable to the model repository id, for example:

```text
Ruaney/geek-genre-text-classifier
```

The Space becomes a small public interface where visitors can test a description.
