"""
System for ranking 5 images by how well they express the target sense
(usually idiomatic vs literal) of a multiword expression, using:

- Dataset layout:
    root/
      dev/
        <compound>/
          <image_id>.png
        subtask_a_dev.tsv
      train/
        ...
      test/
        ...

- TSV columns:
    compound	subset	sentence_type	sentence	expected_order
    image1_name	image1_caption
    ...
    image5_name	image5_caption

- Multilingual-safe text embeddings (SentenceTransformers)
- CLIP image-text similarity
- OpenAI LLM scoring (gpt-4.1-mini) for semantic fit

You can run this script on 'dev' or 'train' to generate predicted orders
and compare with expected_order (e.g., Spearman / accuracy).
"""

import os
import csv
import ast
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from sentence_transformers import SentenceTransformer
import open_clip
from openai import OpenAI

# -----------------------------
# Data classes
# -----------------------------

@dataclass
class RowItem:
    compound: str
    subset: str
    sentence_type: str       # "idiomatic" / "literal" / etc.
    sentence: str
    expected_order: Optional[List[str]]  # list of image_ids or None (for test)
    image_names: List[str]
    image_captions: List[str]


@dataclass
class ImageScore:
    compound: str
    split: str
    sentence_type: str
    sentence: str
    image_name: str
    emb_text_score: float
    emb_clip_score: float
    emb_combined_score: float
    llm_score_0_100: Optional[float]
    final_score: float
    rank: int


# -----------------------------
# Loader for TSV + file layout
# -----------------------------

def normalize_compound_for_filesystem(compound: str) -> str:
    """
    Convert compound name from TSV to filesystem-safe directory name.
    The ' character is replaced with _ on the filesystem.
    """
    return compound.replace("'", "_")


def load_split_tsv(tsv_path: str) -> List[RowItem]:
    rows: List[RowItem] = []

    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            compound = r["compound"]
            subset = r.get("subset", "unknown")
            sentence_type = r.get("sentence_type", "unknown")
            sentence = r["sentence"]
            expected_order_raw = r.get("expected_order", "").strip()

            if expected_order_raw:
                try:
                    expected_order = ast.literal_eval(expected_order_raw)
                    if not isinstance(expected_order, list):
                        expected_order = None
                except Exception:
                    expected_order = None
            else:
                expected_order = None

            image_names = []
            image_captions = []
            for i in range(1, 6):
                name_col = f"image{i}_name"
                cap_col = f"image{i}_caption"
                img_name = r[name_col].strip()
                img_cap = r[cap_col]
                image_names.append(img_name)
                image_captions.append(img_cap)

            rows.append(
                RowItem(
                    compound=compound,
                    subset=subset,
                    sentence_type=sentence_type,
                    sentence=sentence,
                    expected_order=expected_order,
                    image_names=image_names,
                    image_captions=image_captions,
                )
            )

    return rows


# -----------------------------
# Core evaluator
# -----------------------------

class IdiomImageRanker:
    def __init__(
        self,
        text_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        clip_model_name: str = "ViT-B-32",
        clip_pretrained: str = "laion2b_s34b_b79k",
        openai_model: str = "gpt-5.1-mini",
        weight_text_vs_clip: float = 0.6,
        weight_embeddings_vs_llm: float = 0.4,
        device: Optional[str] = None,
    ):
        """
        weight_text_vs_clip: weight of caption-gloss text similarity vs image-gloss CLIP sim
        weight_embeddings_vs_llm: weight of embeddings vs LLM score in final score
        """
        self.client = OpenAI()

        self.text_model = SentenceTransformer(text_model_name)
        self.text_model.max_seq_length = 128

        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained
        )
        self.clip_tokenizer = open_clip.get_tokenizer(clip_model_name)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model.to(self.device)
        self.clip_model.eval()

        self.openai_model = openai_model
        self.weight_text_vs_clip = weight_text_vs_clip
        self.weight_embeddings_vs_llm = weight_embeddings_vs_llm

        # Cache glosses per (compound, sentence_type)
        self.gloss_cache: Dict[Tuple[str, str], str] = {}

    # ---------- Embeddings ----------

    def _embed_text(self, texts: List[str]) -> np.ndarray:
        emb = self.text_model.encode(texts, normalize_embeddings=True)
        return np.asarray(emb)

    def _embed_images(self, image_paths: List[str]) -> np.ndarray:
        imgs = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(self.clip_preprocess(img))
        batch = torch.stack(imgs).to(self.device)

        with torch.no_grad():
            feats = self.clip_model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    def _embed_clip_text(self, texts: List[str]) -> np.ndarray:
        with torch.no_grad():
            tokens = self.clip_tokenizer(texts).to(self.device)
            feats = self.clip_model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b.T

    # ---------- Helper to extract text from API response ----------

    @staticmethod
    def _extract_response_text(resp) -> str:
        """
        Extract text from OpenAI API response.
        Handles both GPT-4 and GPT-5 response formats.
        """
        output_item = resp.output[0]
        item_type = type(output_item).__name__

        # GPT-5 ResponseReasoningItem
        if item_type == 'ResponseReasoningItem':
            # Try to get the summary
            if hasattr(output_item, 'summary'):
                summary = output_item.summary
                if summary:
                    # Summary might be a string or an object
                    if isinstance(summary, str):
                        return summary.strip()
                    elif hasattr(summary, 'text'):
                        return summary.text.strip()
                    elif hasattr(summary, 'content'):
                        # Summary might have content array
                        if isinstance(summary.content, list):
                            return summary.content[0].text.strip()
                        return str(summary.content).strip()
                    else:
                        return str(summary).strip()

            # If summary didn't work, check model_dump or to_dict
            if hasattr(output_item, 'model_dump'):
                data = output_item.model_dump()
                if 'summary' in data and data['summary']:
                    summary_data = data['summary']
                    if isinstance(summary_data, dict) and 'content' in summary_data:
                        content = summary_data['content']
                        if isinstance(content, list) and len(content) > 0:
                            return content[0].get('text', str(content[0])).strip()
                    return str(summary_data).strip()

        # GPT-4 ResponseContentItem - direct content access
        if hasattr(output_item, 'content'):
            return output_item.content[0].text.strip()

        # If nothing worked, provide detailed error with actual values
        raise ValueError(
            f"Cannot extract text from response.\n"
            f"Type: {item_type}\n"
            f"Summary value: {getattr(output_item, 'summary', 'N/A')}\n"
            f"Has encrypted_content: {hasattr(output_item, 'encrypted_content')}"
        )

    # ---------- Sentence type inference (LLM) ----------

    def infer_sentence_type(
        self,
        compound: str,
        sentence: str,
        language: str = "en",
    ) -> str:
        """
        Infer whether the compound is used idiomatically or literally
        in the given sentence context.
        Returns: "idiomatic" or "literal"
        """
        system_msg = (
            "You are a linguist specializing in multiword expressions. "
            "Given a multiword expression and a sentence where it appears, "
            "determine if the expression is used IDIOMATICALLY (figuratively, with a non-literal meaning) "
            "or LITERALLY (with its direct, compositional meaning). "
            "Answer with ONLY one word: 'idiomatic' or 'literal'."
        )

        user_msg = f"""
Language: {language}

Expression: "{compound}"

Sentence:
"{sentence}"

Is the expression "{compound}" used idiomatically or literally in this sentence?
Answer with only one word: idiomatic or literal.
"""

        resp = self.client.responses.create(
            model=self.openai_model,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        result = self._extract_response_text(resp).lower().strip()

        # Extract the word "idiomatic" or "literal" from the response
        if "idiomatic" in result:
            return "idiomatic"
        elif "literal" in result:
            return "literal"
        else:
            # Default to idiomatic if unclear
            return "idiomatic"

    # ---------- Gloss generation (LLM) ----------

    def get_gloss(
        self,
        compound: str,
        sentence_type: str,
        language: str = "en",
    ) -> str:
        """
        Get or generate a short gloss (root meaning) for this compound
        in the given sentence_type (idiomatic / literal).
        Cached per (compound, sentence_type).
        """
        key = (compound, sentence_type)
        if key in self.gloss_cache:
            return self.gloss_cache[key]

        system_msg = (
            "You are a lexicographer. "
            "Given a multiword expression and a label like 'idiomatic' or 'literal', "
            "produce a short, clear definition of that SENSE in the requested language. "
            "Maximum 25 words, no examples, just a definition."
        )

        user_msg = f"""
Expression (multiword): "{compound}"
Sense label: "{sentence_type}"
Target language: {language}

Give a short definition of this expression in this sense, in the target language.
"""

        resp = self.client.responses.create(
            model=self.openai_model,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        gloss = self._extract_response_text(resp)
        self.gloss_cache[key] = gloss
        return gloss

    # ---------- LLM scoring of caption vs gloss ----------

    def llm_score_caption(
        self,
        compound: str,
        sentence_type: str,
        gloss: str,
        sentence: str,
        caption: str,
        language: str = "en",
    ) -> float:
        """
        Ask the LLM: how well does this image caption express the target sense
        of the expression in the context of the sentence?
        Return 0..100.
        """
        system_msg = (
            "You are a linguist evaluating how well an IMAGE CAPTION fits a specific SENSE "
            "of a multiword expression in a given sentence. "
            "You only score the semantic fit to that SENSE. "
            "Answer with a single integer from 0 to 100."
        )

        user_msg = f"""
Language: {language}

Expression: "{compound}"
Sense type label: "{sentence_type}"
Sense definition: "{gloss}"

Sentence (context where the expression appears):
"{sentence}"

Image caption:
"{caption}"

Task:
On a scale from 0 (not related at all) to 100 (perfect match),
how well does this IMAGE CAPTION express the TARGET SENSE of the expression
in this sentence?

Answer with a single integer only.
"""

        resp = self.client.responses.create(
            model=self.openai_model,
            input=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        )
        text = self._extract_response_text(resp)

        import re
        m = re.search(r"(\d+)", text)
        if not m:
            return 0.0
        val = int(m.group(1))
        val = max(0, min(100, val))
        return float(val)

    # ---------- Per-row scoring & ranking ----------

    def rank_row(
        self,
        split_root: str,
        split_name: str,
        row: RowItem,
        language: str = "en",
        use_llm: bool = True,
        infer_sentence_type: bool = False,
        normalize_compound: bool = True,
    ) -> List[ImageScore]:
        """
        For a single TSV row, compute scores for its 5 images and return ranked list.

        Args:
            infer_sentence_type: If True, use LLM to infer whether the expression
                                 is used idiomatically or literally from the sentence context.
                                 If False, use the sentence_type from the row data.
            normalize_compound: If True, normalize compound name for filesystem (e.g., ' -> _).
                                If False, use compound name as-is (needed for Uzbek).
        """
        compound = row.compound
        sentence = row.sentence

        # Infer sentence type if requested, otherwise use provided type
        if infer_sentence_type:
            sentence_type = self.infer_sentence_type(compound, sentence, language=language)
            print(f"Inferred sentence type for '{compound}': {sentence_type}")
        else:
            sentence_type = row.sentence_type

        # Normalize compound name for filesystem (e.g., ' -> _)
        # Uzbek keeps the ' character, so we skip normalization for that language
        if normalize_compound:
            compound_dir = normalize_compound_for_filesystem(compound)
        else:
            compound_dir = compound

        image_paths = [
            os.path.join(split_root, compound_dir, image_name)
            for image_name in row.image_names
        ]

        captions = row.image_captions

        gloss = self.get_gloss(compound, sentence_type, language=language)

        prompt_emb = self._embed_text(captions)             # (5, d)
        gloss_emb_text = self._embed_text([gloss])          # (1, d)

        img_emb_clip = self._embed_images(image_paths)      # (5, d_clip)
        gloss_emb_clip = self._embed_clip_text([gloss])     # (1, d_clip)

        sim_text = self._cosine_sim(prompt_emb, gloss_emb_text).reshape(-1)   # (5,)
        sim_clip = self._cosine_sim(img_emb_clip, gloss_emb_clip).reshape(-1) # (5,)

        alpha = self.weight_text_vs_clip
        emb_combined = alpha * sim_text + (1.0 - alpha) * sim_clip           # (5,)

        llm_scores: List[Optional[float]] = [None] * 5
        if use_llm:
            for i, cap in enumerate(captions):
                llm_scores[i] = self.llm_score_caption(
                    compound=compound,
                    sentence_type=sentence_type,
                    gloss=gloss,
                    sentence=sentence,
                    caption=cap,
                    language=language,
                )

        final_scores = []
        for i in range(5):
            if llm_scores[i] is None:
                final_scores.append(emb_combined[i])
            else:
                w = self.weight_embeddings_vs_llm
                final_scores.append(
                    w * emb_combined[i] + (1.0 - w) * (llm_scores[i] / 100.0)
                )
        final_scores = np.asarray(final_scores)

        indices_sorted = np.argsort(-final_scores)  # desc

        results: List[ImageScore] = []
        for rank_pos, idx in enumerate(indices_sorted, start=1):
            results.append(
                ImageScore(
                    compound=compound,
                    split=split_name,
                    sentence_type=sentence_type,
                    sentence=sentence,
                    image_name=row.image_names[idx],
                    emb_text_score=float(sim_text[idx]),
                    emb_clip_score=float(sim_clip[idx]),
                    emb_combined_score=float(emb_combined[idx]),
                    llm_score_0_100=llm_scores[idx],
                    final_score=float(final_scores[idx]),
                    rank=rank_pos,
                )
            )

        return results


# -----------------------------
# Evaluation helpers
# -----------------------------

def order_from_scores(scores: List[ImageScore]) -> List[str]:
    scores_sorted = sorted(scores, key=lambda s: s.rank)
    return [s.image_name for s in scores_sorted]


def compare_orders(pred: List[str], gold: Optional[List[str]]) -> float:
    """
    Simple metric: proportion of images in the same top-1 position.
    You can replace with Spearman/Kendall later as needed.
    """
    if not gold or len(pred) == 0 or len(gold) == 0:
        return 0.0
    return 1.0 if pred[0] == gold[0] else 0.0


# -----------------------------
# Test data processing
# -----------------------------

# Language name to language code mapping
LANGUAGE_CODE_MAP = {
    "Chinese": "ZH",
    "Georgian": "KA",
    "Greek": "EL",
    "Igbo": "IG",
    "Kazakh": "KK",
    "Norwegian": "NO",
    "Portuguese-Brazil": "PT-BR",
    "Portuguese-Portugal": "PT-PT",
    "Russian": "RU",
    "Serbian": "SR",
    "Slovak": "SK",
    "Slovenian": "SL",
    "Spanish-Ecuador": "ES-EC",
    "Turkish": "TR",
    "Uzbek": "UZ",
}

# Language name to ISO 639-1/639-3 code mapping for model language parameter
LANGUAGE_ISO_MAP = {
    "Chinese": "zh",
    "Georgian": "ka",
    "Greek": "el",
    "Igbo": "ig",
    "Kazakh": "kk",
    "Norwegian": "no",
    "Portuguese-Brazil": "pt",
    "Portuguese-Portugal": "pt",
    "Russian": "ru",
    "Serbian": "sr",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Spanish-Ecuador": "es",
    "Turkish": "tr",
    "Uzbek": "uz",
}


def process_test_language(
    language_name: str,
    test_root: str,
    ranker: "IdiomImageRanker",
    output_dir: str,
) -> str:
    """
    Process a single language from the test set and generate submission TSV.

    Args:
        language_name: Full language name (e.g., "Chinese", "Georgian")
        test_root: Root directory of test data (contains language folders and admire_file)
        ranker: IdiomImageRanker instance
        output_dir: Directory to save submission files

    Returns:
        Path to generated submission file
    """
    print(f"\n{'='*60}")
    print(f"Processing language: {language_name}")
    print(f"{'='*60}")

    # Load TSV file
    tsv_path = os.path.join(test_root, "admire_file", f"submission_{language_name}.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    rows = load_split_tsv(tsv_path)
    print(f"Loaded {len(rows)} rows from {tsv_path}")

    # Get language code and ISO code
    lang_code = LANGUAGE_CODE_MAP.get(language_name, language_name)
    lang_iso = LANGUAGE_ISO_MAP.get(language_name, "en")

    # Path to images
    images_root = os.path.join(test_root, language_name)

    # Uzbek uses ' in folder names, other languages use _ instead
    normalize_compound = (language_name != "Uzbek")

    # Process each row and collect results
    results = []
    for idx, row in enumerate(rows, 1):
        print(f"Processing {idx}/{len(rows)}: {row.compound}")

        # Rank images for this row
        row_scores = ranker.rank_row(
            split_root=images_root,
            split_name="test",
            row=row,
            language=lang_iso,
            use_llm=True,
            infer_sentence_type=True,  # Always infer for test data
            normalize_compound=normalize_compound,  # Don't normalize for Uzbek
        )

        # Get predicted order
        predicted_order = order_from_scores(row_scores)

        # Store result
        results.append({
            "compound": row.compound,
            "sentence": row.sentence,
            "expected_order": predicted_order,
        })

    # Write submission file
    output_filename = f"submission_{lang_code}.tsv"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["compound", "expected_order", "sentence"])

        for result in results:
            # Format expected_order as Python list string
            order_str = str(result["expected_order"])
            writer.writerow([result["compound"], order_str, result["sentence"]])

    print(f"Saved submission file: {output_path}")
    return output_path


# -----------------------------
# Main example usage
# -----------------------------

if __name__ == "__main__":
    # -----------------------------
    # Configuration
    # -----------------------------

    # MODE: "dev" for development/training, "test" for generating test submissions
    MODE = "test"  # Change to "dev" for training/validation

    ROOT_DIR = "./"  # contains dev/train/test

    # Initialize ranker
    ranker = IdiomImageRanker(
        weight_text_vs_clip=0.6,
        weight_embeddings_vs_llm=0.4,
        #openai_model="gpt-4.1-mini",
        openai_model="gpt-5.1",
    )

    # -----------------------------
    # TEST MODE - Generate submissions for all languages
    # -----------------------------
    if MODE == "test":
        TEST_ROOT = os.path.join(ROOT_DIR, "test")
        OUTPUT_DIR = os.path.join(ROOT_DIR, "submissions")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Specify which languages to process (or process all)
        # Leave empty to process all languages
        LANGUAGES_TO_PROCESS = []  # Process all languages

        if not LANGUAGES_TO_PROCESS:
            LANGUAGES_TO_PROCESS = list(LANGUAGE_CODE_MAP.keys())

        print(f"Processing {len(LANGUAGES_TO_PROCESS)} languages...")
        print(f"Languages: {', '.join(LANGUAGES_TO_PROCESS)}")

        for language in LANGUAGES_TO_PROCESS:
            try:
                output_path = process_test_language(
                    language_name=language,
                    test_root=TEST_ROOT,
                    ranker=ranker,
                    output_dir=OUTPUT_DIR,
                )
                print(f"✓ Successfully generated: {output_path}")
            except Exception as e:
                print(f"✗ Error processing {language}: {e}")
                import traceback
                traceback.print_exc()

        print(f"\n{'='*60}")
        print(f"All submission files saved to: {OUTPUT_DIR}")
        print(f"{'='*60}")

    # -----------------------------
    # DEV MODE - Training/validation evaluation
    # -----------------------------
    elif MODE == "dev":
        SPLIT = "dev"  # "dev" or "train"
        TSV_NAME = f"subtask_a_{SPLIT}.tsv"
        INFER_SENTENCE_TYPE = False  # Use ground truth sentence_type for dev/train

        split_root = os.path.join(ROOT_DIR, SPLIT)
        tsv_path = os.path.join(ROOT_DIR, SPLIT, TSV_NAME)

        rows = load_split_tsv(tsv_path)

        all_scores: List[ImageScore] = []
        top1_correct = 0
        total = 0

        for row in rows:
            row_scores = ranker.rank_row(
                split_root=split_root,
                split_name=SPLIT,
                row=row,
                language="en",
                use_llm=True,
                infer_sentence_type=INFER_SENTENCE_TYPE,
            )
            all_scores.extend(row_scores)

            pred_order = order_from_scores(row_scores)
            acc = compare_orders(pred_order, row.expected_order)
            if row.expected_order:
                top1_correct += acc
                total += 1

        if total > 0:
            print(f"Top-1 accuracy on {SPLIT}: {top1_correct / total:.3f}")
        else:
            print(f"No expected_order available for {SPLIT}; only predictions computed.")

        # Write detailed predictions to TSV
        out_path = os.path.join(ROOT_DIR, f"predictions_{SPLIT}.tsv")
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(
                [
                    "compound",
                    "split",
                    "sentence_type",
                    "sentence",
                    "image_name",
                    "rank",
                    "emb_text_score",
                    "emb_clip_score",
                    "emb_combined_score",
                    "llm_score_0_100",
                    "final_score",
                ]
            )
            for s in all_scores:
                writer.writerow(
                    [
                        s.compound,
                        s.split,
                        s.sentence_type,
                        s.sentence,
                        s.image_name,
                        s.rank,
                        f"{s.emb_text_score:.6f}",
                        f"{s.emb_clip_score:.6f}",
                        f"{s.emb_combined_score:.6f}",
                        "" if s.llm_score_0_100 is None else f"{s.llm_score_0_100:.2f}",
                        f"{s.final_score:.6f}",
                    ]
                )

        print(f"Saved predictions to: {out_path}")
