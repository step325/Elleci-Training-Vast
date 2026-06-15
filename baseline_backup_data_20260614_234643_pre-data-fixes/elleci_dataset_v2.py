"""
Elleci v2 Dataset - EN-only pretraining (top-tier mix, max intelligence density)

Strategia: pretraining 100% EN con SOLO i dataset top di ogni categoria.
Italiano via fine-tuning separato post-training (Llamantino-style).

Phase 1: EN Foundation web (60% del budget totale)
    - FineWeb-Edu: 45% (1.3T, edu-filtered, top intelligence/token)
    - DCLM-Baseline: 25% (4T, top diversity baseline)
    - Nemotron-CC v2: 20% (NVIDIA HQ subset, +5.6 MMLU vs DCLM)
    - Cosmopedia V2: 10% (synthetic textbook, knowledge density)

Phase 2: Reasoning heavy - math + code (25%)
    - The Stack v2 smol-ids: 30% (code repo-level, 17 lang core)
    - Nemotron-CC-Math 4+: 25% (NVIDIA 52B math HQ, Lynx+LLM pipeline)
    - FineMath-4+: 20% (HF math reasoning HQ 9.6B)
    - Proof-Pile-2: 15% (formal math + arxiv + algebraic stack 55B)
    - MathCode-Pile: 10% (math+code intersezione reasoning steps 19B)

Phase 3: Reasoning + Instruction (15%)
    - OpenR1-Math-220k: 25% (220K DeepSeek-R1 CoT verificati, top math reasoning)
    - OpenMathReasoning: 25% (NVIDIA 306K AoPS, R1/QwQ solutions, no gate)
    - Natural Reasoning: 20% (Meta 1.15M general reasoning, CC-BY-NC)
    - Tulu 3 SFT mix: 20% (Allen AI, top EN instruction 2025)
    - Nemotron-Pretraining-SFT v1: 10% (NVIDIA STEM/code/math SFT)

Data Sources (HF datasets):
- FineWeb-Edu: HuggingFaceFW/fineweb-edu (ODC-BY, no gate)
- DCLM-Baseline: mlfoundations/dclm-baseline-1.0 (CC-BY-4.0, no gate)
- Nemotron-CC v2: nvidia/Nemotron-CC-v2 (NVIDIA agreement, GATED)
- Cosmopedia V2: HuggingFaceTB/smollm-corpus, config cosmopedia-v2 (ODC-BY)
- The Stack v2: bigcode/the-stack-v2-train-smol-ids (TOS accept, GATED)
- Nemotron-CC-Math: nvidia/Nemotron-CC-Math-v1 "4plus" (GATED)
- FineMath: HuggingFaceTB/finemath "finemath-4plus" (ODC-By, no gate)
- Proof-Pile-2: EleutherAI/proof-pile-2 (mix licenses, no gate)
- MathCode-Pile: MathGenie/MathCode-Pile (Apache 2.0, no gate)
- OpenR1-Math-220k: open-r1/OpenR1-Math-220k "default" (Apache 2.0, no gate)
- OpenMathReasoning: nvidia/OpenMathReasoning (CC-BY-4.0, no gate!)
- Natural Reasoning: facebook/natural_reasoning (CC-BY-NC-4.0, no gate)
- Tulu 3: allenai/tulu-3-sft-mixture (ODC-BY-1.0, no gate)
- Nemotron-Pretraining-SFT: nvidia/Nemotron-Pretraining-SFT-v1 (GATED)

GATED DATASETS — RICHIEDONO ACCEPT TOS SU HF WEB PRIMA DI USARE:
  1. nvidia/Nemotron-CC-v2          → https://huggingface.co/datasets/nvidia/Nemotron-CC-v2
  2. nvidia/Nemotron-CC-Math-v1     → https://huggingface.co/datasets/nvidia/Nemotron-CC-Math-v1
  3. nvidia/Nemotron-Pretraining-SFT-v1 → https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-SFT-v1
  4. bigcode/the-stack-v2-train-smol-ids → https://huggingface.co/datasets/bigcode/the-stack-v2-train-smol-ids

NOTA: I loader IT/legacy (CulturaX, Wikipedia IT, Fauno, Alpaca-IT, Dolly, OpenOrca,
OpenHermes, MetaMath, Magpie, CodeAlpaca, OpenWebMath, StarCoder PR) restano nel file
per riuso in fase POST-training (continued pretraining IT + SFT IT).
"""
import torch
from torch.utils.data import IterableDataset
import random
from datasets import load_dataset
import json
import os
import glob
from typing import Optional, Dict, Iterator, List


class EllediDatasetV2(IterableDataset):
    """
    Streaming dataset for Elleci v2 3-phase training.

    Args:
        tokenizer: Tokenizer instance
        phase: Training phase (1, 2, or 3)
        max_length: Maximum sequence length
        batch_size: Batch size for collation
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        tokenizer,
        phase: int = 1,
        max_length: int = 512,
        batch_size: int = 4,
        seed: int = 42,
        hf_token: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.phase = phase
        self.max_length = max_length
        self.batch_size = batch_size
        self.seed = seed
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

        # Define mixing ratios based on phase
        self._setup_ratios()

        print(f"Elleci v2 Dataset - Phase {phase}")
        print(f"  Max Length: {max_length}, Batch Size: {batch_size}")
        print(f"  Ratios: {self.ratios}")

        # Load local instructions (fast, in-memory)
        self.local_instructions = self._load_local_instructions()
        print(f"  Local instructions loaded: {len(self.local_instructions)}")

        # Streaming iterators (created lazily)
        self._iterators: Dict[str, Optional[Iterator]] = {}
        self._init_streams()

    def _setup_ratios(self):
        """Set up data mixing ratios based on training phase (top-tier EN mix)."""
        if self.phase == 1:
            # Phase 1: EN Foundation web — solo top-tier
            self.ratios = {
                'fineweb_edu': 0.45,
                'dclm_baseline': 0.25,
                'nemotron_cc_v2': 0.20,
                'cosmopedia': 0.10,
            }
            self.sources = list(self.ratios.keys())
        elif self.phase == 2:
            # Phase 2: Math + code — solo top-tier
            self.ratios = {
                'stack_v2_smol': 0.30,
                'nemotron_cc_math': 0.25,
                'finemath_4plus': 0.20,
                'proof_pile_2': 0.15,
                'mathcode_pile': 0.10,
            }
            self.sources = list(self.ratios.keys())
        elif self.phase == 3:
            # Phase 3: Reasoning + instruction — solo top-tier
            self.ratios = {
                'openr1_math': 0.25,        # DeepSeek-R1 CoT verificati 220K
                'openmath_reasoning': 0.25, # NVIDIA AoPS 306K, no gate
                'natural_reasoning': 0.20,  # Meta 1.15M general reasoning
                'tulu3': 0.20,              # Allen AI top instruction
                'nemotron_sft_v1': 0.10,    # NVIDIA STEM/code/math SFT
            }
            self.sources = list(self.ratios.keys())
        else:
            raise ValueError(f"Unknown phase: {self.phase}")

    def _init_streams(self):
        """Initialize all stream iterators to None (lazy loading)."""
        for source in self.sources:
            self._iterators[source] = None

    def _load_local_instructions(self) -> List[dict]:
        """Load Italian instructions from local JSONL files."""
        instructions = []

        # Try instruction files
        files = glob.glob("data/elleci_instructions_final.jsonl")
        if not files:
            files = glob.glob("data/elleci_instructions.jsonl")

        for fpath in files:
            if os.path.exists(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                data = json.loads(line)
                                instructions.append(data)
                            except json.JSONDecodeError:
                                pass

        return instructions

    def _format_chatml(self, user: str, assistant: str) -> str:
        """Format instruction as ChatML."""
        return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"

    def _format_alpaca(self, instruction: str, input_text: str, output: str) -> str:
        """Format Alpaca-style instruction."""
        if input_text:
            user = f"{instruction}\n\n{input_text}"
        else:
            user = instruction
        return self._format_chatml(user, output)

    # ========== Stream Loaders ==========

    def _get_fineweb_edu_stream(self) -> Iterator:
        """FineWeb-Edu: High-quality educational web content."""
        print("Loading FineWeb-Edu stream...")
        try:
            ds = load_dataset(
                "HuggingFaceFW/fineweb-edu",
                name="sample-10BT",  # Use 10B token sample for manageability
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=1000))
        except Exception as e:
            print(f"  FineWeb-Edu failed: {e}, falling back to Cosmopedia")
            return self._get_cosmopedia_stream()

    def _get_cosmopedia_stream(self) -> Iterator:
        """Cosmopedia V2: Synthetic educational content."""
        print("Loading Cosmopedia V2 stream...")
        try:
            ds = load_dataset(
                "HuggingFaceTB/smollm-corpus",
                "cosmopedia-v2",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=1000))
        except Exception as e:
            print(f"  Cosmopedia failed: {e}")
            raise

    def _get_openwebmath_stream(self) -> Iterator:
        """OpenWebMath: Mathematics content."""
        print("Loading OpenWebMath stream...")
        try:
            ds = load_dataset(
                "open-web-math/open-web-math",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=1000))
        except Exception as e:
            print(f"  OpenWebMath failed: {e}, falling back to Cosmopedia")
            return self._get_cosmopedia_stream()

    def _get_stack_stream(self) -> Iterator:
        """The Stack v2: Code content."""
        print("Loading The Stack v2 stream...")
        try:
            ds = load_dataset(
                "bigcode/starcoderdata",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=1000))
        except Exception as e:
            print(f"  The Stack failed: {e}, falling back to Cosmopedia")
            return self._get_cosmopedia_stream()

    def _get_culturax_it_stream(self) -> Iterator:
        """CulturaX Italian: High-quality Italian web content."""
        print("Loading CulturaX Italian stream...")
        try:
            ds = load_dataset(
                "uonlp/CulturaX",
                "it",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=1000))
        except Exception as e:
            print(f"  CulturaX failed: {e}, trying Wikipedia IT")
            return self._get_wikipedia_it_stream()

    def _get_wikipedia_it_stream(self) -> Iterator:
        """Wikipedia IT: Italian encyclopedia."""
        print("Loading Wikipedia IT stream...")
        try:
            ds = load_dataset(
                "wikimedia/wikipedia",
                "20231101.it",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=1000))
        except Exception as e:
            print(f"  Wikipedia IT failed: {e}")
            # Try alternative
            try:
                ds = load_dataset(
                    "graelo/wikipedia",
                    "20230601.it",
                    split="train",
                    streaming=True,
                    token=self.hf_token
                )
                return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=1000))
            except Exception as e2:
                print(f"  Alternative Wikipedia also failed: {e2}")
                raise

    def _get_openorca_stream(self) -> Iterator:
        """OpenOrca: Reasoning and math instructions."""
        print("Loading OpenOrca stream...")
        try:
            ds = load_dataset(
                "Open-Orca/OpenOrca",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=100))
        except Exception as e:
            print(f"  OpenOrca failed: {e}")
            return iter([])

    def _get_fauno_it_stream(self) -> Iterator:
        """Fauno IT: Italian Q&A (StackOverflow + Quora)."""
        print("Loading Fauno IT streams...")
        try:
            # Try StackOverflow first
            ds = load_dataset(
                "andreabac3/StackOverflow-Italian-Fauno-Baize",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=500))
        except Exception as e:
            print(f"  Fauno StackOverflow failed: {e}, trying Quora")
            try:
                ds = load_dataset(
                    "andreabac3/Quora-Italian-Fauno-Baize",
                    split="train",
                    streaming=True,
                    token=self.hf_token
                )
                return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=500))
            except Exception as e2:
                print(f"  Fauno Quora also failed: {e2}")
                return iter([])

    def _get_alpaca_it_stream(self) -> Iterator:
        """Alpaca IT (Camoscio): Italian instructions."""
        print("Loading Alpaca IT (Camoscio) stream...")
        try:
            ds = load_dataset(
                "teelinsan/camoscio",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=500))
        except Exception as e:
            print(f"  Alpaca IT failed: {e}")
            return iter([])

    def _get_dolly_stream(self) -> Iterator:
        """Dolly: Diverse English instructions."""
        print("Loading Dolly stream...")
        try:
            ds = load_dataset(
                "databricks/databricks-dolly-15k",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=500))
        except Exception as e:
            print(f"  Dolly failed: {e}")
            return iter([])

    def _get_codealpaca_stream(self) -> Iterator:
        """CodeAlpaca: Code instructions."""
        print("Loading CodeAlpaca stream...")
        try:
            ds = load_dataset(
                "sahil2801/CodeAlpaca-20k",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=500))
        except Exception as e:
            print(f"  CodeAlpaca failed: {e}")
            return iter([])

    # ========== EN-only Pretraining Streams (Phase 1/2/3 mix) ==========

    def _get_dclm_baseline_stream(self) -> Iterator:
        """DCLM-Baseline: 3.8T fastText-filtered Common Crawl."""
        print("Loading DCLM-Baseline stream...")
        try:
            ds = load_dataset(
                "mlfoundations/dclm-baseline-1.0",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=1000))
        except Exception as e:
            print(f"  DCLM-Baseline failed: {e}, falling back to FineWeb-Edu")
            return self._get_fineweb_edu_stream()

    def _get_nemotron_cc_v2_stream(self) -> Iterator:
        """Nemotron-CC v2: NVIDIA HQ Common Crawl subset."""
        print("Loading Nemotron-CC v2 stream...")
        try:
            ds = load_dataset(
                "nvidia/Nemotron-CC-v2",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=1000))
        except Exception as e:
            print(f"  Nemotron-CC v2 failed: {e}, falling back to FineWeb-Edu")
            return self._get_fineweb_edu_stream()

    def _get_nemotron_cc_math_stream(self) -> Iterator:
        """Nemotron-CC-Math 4+: NVIDIA HQ math subset (133B token)."""
        print("Loading Nemotron-CC-Math 4+ stream...")
        try:
            ds = load_dataset(
                "nvidia/Nemotron-CC-Math-v1",
                "4plus",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=500))
        except Exception as e:
            print(f"  Nemotron-CC-Math failed: {e}, falling back to OpenWebMath")
            return self._get_openwebmath_stream()

    def _get_finemath_4plus_stream(self) -> Iterator:
        """FineMath 4+: HF math reasoning HQ (9.6B token, 6.7M docs)."""
        print("Loading FineMath 4+ stream...")
        try:
            ds = load_dataset(
                "HuggingFaceTB/finemath",
                "finemath-4plus",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=500))
        except Exception as e:
            print(f"  FineMath 4+ failed: {e}, falling back to OpenWebMath")
            return self._get_openwebmath_stream()

    def _get_proof_pile_2_stream(self) -> Iterator:
        """Proof-Pile-2: formal math + arxiv math + algebraic stack."""
        print("Loading Proof-Pile-2 stream...")
        try:
            ds = load_dataset(
                "EleutherAI/proof-pile-2",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=500))
        except Exception as e:
            print(f"  Proof-Pile-2 failed: {e}, falling back to OpenWebMath")
            return self._get_openwebmath_stream()

    def _get_mathcode_pile_stream(self) -> Iterator:
        """MathCode-Pile: math+code intersezione con reasoning steps (19.2B)."""
        print("Loading MathCode-Pile stream...")
        try:
            ds = load_dataset(
                "MathGenie/MathCode-Pile",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=500))
        except Exception as e:
            print(f"  MathCode-Pile failed: {e}, falling back to OpenWebMath")
            return self._get_openwebmath_stream()

    def _get_stack_v2_smol_stream(self) -> Iterator:
        """The Stack v2 smol-ids: code repo-level (17 lang core, StarCoder2 train)."""
        print("Loading The Stack v2 smol-ids stream...")
        try:
            ds = load_dataset(
                "bigcode/the-stack-v2-train-smol-ids",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=500))
        except Exception as e:
            print(f"  Stack v2 smol failed: {e}, falling back to starcoderdata")
            return self._get_stack_stream()

    def _get_starcoder_pr_stream(self) -> Iterator:
        """StarCoder PRs: GitHub PR + Jupyter + Kaggle for code diversity."""
        print("Loading StarCoder GitHub PRs stream...")
        try:
            ds = load_dataset(
                "bigcode/starcoder2data-extras",
                "github-issues",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=500))
        except Exception as e:
            print(f"  StarCoder PRs failed: {e}, falling back to starcoderdata")
            return self._get_stack_stream()

    def _get_tulu3_stream(self) -> Iterator:
        """Tulu 3 SFT mixture: best EN instruction 2025 (Allen AI)."""
        print("Loading Tulu 3 SFT mixture stream...")
        try:
            ds = load_dataset(
                "allenai/tulu-3-sft-mixture",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=500))
        except Exception as e:
            print(f"  Tulu 3 failed: {e}, falling back to OpenHermes")
            return self._get_openhermes_stream()

    def _get_openhermes_stream(self) -> Iterator:
        """OpenHermes 2.5: diverse general instruction (~1M conv)."""
        print("Loading OpenHermes 2.5 stream...")
        try:
            ds = load_dataset(
                "teknium/OpenHermes-2.5",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=500))
        except Exception as e:
            print(f"  OpenHermes 2.5 failed: {e}")
            return iter([])

    def _get_metamath_stream(self) -> Iterator:
        """MetaMathQA: math instruction (GSM8K-style augmentation)."""
        print("Loading MetaMathQA stream...")
        try:
            ds = load_dataset(
                "meta-math/MetaMathQA",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=500))
        except Exception as e:
            print(f"  MetaMathQA failed: {e}")
            return iter([])

    def _get_magpie_stream(self) -> Iterator:
        """Magpie Pro: synthetic alignment data (300K multi-turn)."""
        print("Loading Magpie Pro MT 300K stream...")
        try:
            ds = load_dataset(
                "Magpie-Align/Magpie-Pro-MT-300K-v0.1",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=500))
        except Exception as e:
            print(f"  Magpie failed: {e}")
            return iter([])

    def _get_openr1_math_stream(self) -> Iterator:
        """OpenR1-Math-220k default split: DeepSeek-R1 CoT verificati."""
        print("Loading OpenR1-Math-220k default stream...")
        try:
            ds = load_dataset(
                "open-r1/OpenR1-Math-220k",
                "default",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=500))
        except Exception as e:
            print(f"  OpenR1-Math failed: {e}, falling back to MetaMathQA")
            return self._get_metamath_stream()

    def _get_openmath_reasoning_stream(self) -> Iterator:
        """NVIDIA OpenMathReasoning: 306K AoPS problems, R1/QwQ solutions."""
        print("Loading NVIDIA OpenMathReasoning (cot split) stream...")
        try:
            ds = load_dataset(
                "nvidia/OpenMathReasoning",
                split="cot",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=500))
        except Exception as e:
            print(f"  OpenMathReasoning failed: {e}, falling back to OpenR1-Math")
            return self._get_openr1_math_stream()

    def _get_natural_reasoning_stream(self) -> Iterator:
        """Meta NaturalReasoning: 1.15M general reasoning backtranslated."""
        print("Loading Meta NaturalReasoning stream...")
        try:
            ds = load_dataset(
                "facebook/natural_reasoning",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=500))
        except Exception as e:
            print(f"  NaturalReasoning failed: {e}")
            return iter([])

    def _get_nemotron_sft_v1_stream(self) -> Iterator:
        """NVIDIA Nemotron-Pretraining-SFT v1: STEM/code/math SFT."""
        print("Loading Nemotron-Pretraining-SFT v1 stream...")
        try:
            ds = load_dataset(
                "nvidia/Nemotron-Pretraining-SFT-v1",
                split="train",
                streaming=True,
                token=self.hf_token
            )
            return iter(ds.shuffle(seed=random.randint(0, 100000), buffer_size=500))
        except Exception as e:
            print(f"  Nemotron-SFT v1 failed: {e}, falling back to Tulu 3")
            return self._get_tulu3_stream()

    # ========== Sample Getters ==========

    _STREAM_DISPATCH = {
        # Foundation web (EN)
        'fineweb_edu': '_get_fineweb_edu_stream',
        'dclm_baseline': '_get_dclm_baseline_stream',
        'nemotron_cc_v2': '_get_nemotron_cc_v2_stream',
        'cosmopedia': '_get_cosmopedia_stream',
        # Math + code
        'stack_v2_smol': '_get_stack_v2_smol_stream',
        'finemath_4plus': '_get_finemath_4plus_stream',
        'nemotron_cc_math': '_get_nemotron_cc_math_stream',
        'openwebmath': '_get_openwebmath_stream',
        'proof_pile_2': '_get_proof_pile_2_stream',
        'mathcode_pile': '_get_mathcode_pile_stream',
        'starcoder_pr': '_get_starcoder_pr_stream',
        # Reasoning + Instruction (top tier)
        'openr1_math': '_get_openr1_math_stream',
        'openmath_reasoning': '_get_openmath_reasoning_stream',
        'natural_reasoning': '_get_natural_reasoning_stream',
        'tulu3': '_get_tulu3_stream',
        'nemotron_sft_v1': '_get_nemotron_sft_v1_stream',
        # Legacy / fallback sources
        'openhermes': '_get_openhermes_stream',
        'metamath': '_get_metamath_stream',
        'magpie': '_get_magpie_stream',
        'codealpaca': '_get_codealpaca_stream',
        'stack': '_get_stack_stream',
        'culturax_it': '_get_culturax_it_stream',
        'wikipedia_it': '_get_wikipedia_it_stream',
        'english_mix': '_get_fineweb_edu_stream',
        'openorca': '_get_openorca_stream',
        'fauno_it': '_get_fauno_it_stream',
        'alpaca_it': '_get_alpaca_it_stream',
        'dolly': '_get_dolly_stream',
    }

    def _get_stream(self, source: str) -> Iterator:
        """Get or create stream for source."""
        if self._iterators.get(source) is None:
            if source == 'local_it':
                # Local instructions don't need a stream
                return None
            method_name = self._STREAM_DISPATCH.get(source)
            if method_name is None:
                raise ValueError(f"Unknown source: {source}")
            self._iterators[source] = getattr(self, method_name)()

        return self._iterators.get(source)

    def _reset_stream(self, source: str):
        """Reset a stream after StopIteration."""
        self._iterators[source] = None

    def _get_next_sample(self, source: str) -> Optional[str]:
        """Get next text sample from specified source."""
        try:
            # Handle local instructions separately
            if source == 'local_it':
                if not self.local_instructions:
                    return None
                item = random.choice(self.local_instructions)
                inst = item.get("instruction", "")
                inp = item.get("input", "")
                out = item.get("output", "")
                full_inst = f"{inst}\n{inp}".strip() if inp else inst
                return self._format_chatml(full_inst, out)

            # Get stream
            stream = self._get_stream(source)
            if stream is None:
                return None

            item = next(stream)

            # Extract text based on source format
            plain_text_sources = {
                'fineweb_edu', 'dclm_baseline', 'nemotron_cc_v2', 'cosmopedia',
                'stack_v2_smol', 'finemath_4plus', 'nemotron_cc_math',
                'openwebmath', 'proof_pile_2', 'mathcode_pile', 'starcoder_pr',
                'stack', 'culturax_it', 'wikipedia_it', 'english_mix',
            }
            messages_sources = {'tulu3', 'openhermes', 'magpie', 'nemotron_sft_v1'}

            if source in plain_text_sources:
                # HF datasets standard "text" field; some use "content" or "raw_content"
                text = item.get("text") or item.get("content") or item.get("raw_content") or ""
            elif source in messages_sources:
                # ChatML-formatted multi-turn: {"messages": [{"role": ..., "content": ...}, ...]}
                messages = item.get("messages") or item.get("conversations") or []
                parts = []
                for m in messages:
                    role = m.get("role") or m.get("from") or "user"
                    content = m.get("content") or m.get("value") or ""
                    if not content:
                        continue
                    role_norm = "assistant" if role in ("assistant", "gpt", "model") else "user"
                    parts.append(f"<|im_start|>{role_norm}\n{content}<|im_end|>")
                text = "\n".join(parts)
            elif source == 'openr1_math':
                # OpenR1-Math: prefer "messages" field if present, else build from problem+solution
                messages = item.get("messages") or []
                if messages:
                    parts = []
                    for m in messages:
                        role = m.get("role") or "user"
                        content = m.get("content") or ""
                        if not content:
                            continue
                        role_norm = "assistant" if role == "assistant" else "user"
                        parts.append(f"<|im_start|>{role_norm}\n{content}<|im_end|>")
                    text = "\n".join(parts)
                else:
                    problem = item.get("problem", "")
                    solution = item.get("solution", "")
                    text = self._format_chatml(problem, solution) if problem and solution else ""
            elif source == 'openmath_reasoning':
                # NVIDIA OpenMathReasoning: problem + generated_solution
                problem = item.get("problem", "")
                solution = item.get("generated_solution", "")
                text = self._format_chatml(problem, solution) if problem and solution else ""
            elif source == 'natural_reasoning':
                # Meta NaturalReasoning: question + responses[0].response
                question = item.get("question", "")
                responses = item.get("responses") or []
                response = responses[0].get("response", "") if responses else item.get("reference_answer", "")
                text = self._format_chatml(question, response) if question and response else ""
            elif source == 'metamath':
                # MetaMathQA format: {"query": ..., "response": ...}
                query = item.get("query") or item.get("question") or ""
                response = item.get("response") or item.get("answer") or ""
                text = self._format_chatml(query, response) if query and response else ""
            elif source == 'openorca':
                # OpenOrca format
                system = item.get("system_prompt", "")
                question = item.get("question", "")
                response = item.get("response", "")
                user = f"{system}\n\n{question}".strip() if system else question
                text = self._format_chatml(user, response)
            elif source == 'fauno_it':
                # Fauno format: input field contains conversation
                text = item.get("input", "")
            elif source == 'alpaca_it':
                # Alpaca/Camoscio format
                inst = item.get("instruction", "")
                inp = item.get("input", "")
                out = item.get("output", "")
                text = self._format_alpaca(inst, inp, out)
            elif source == 'dolly':
                # Dolly format
                inst = item.get("instruction", "")
                context = item.get("context", "")
                response = item.get("response", "")
                user = f"{inst}\n\n{context}".strip() if context else inst
                text = self._format_chatml(user, response)
            elif source == 'codealpaca':
                # CodeAlpaca format
                inst = item.get("instruction", "")
                inp = item.get("input", "")
                out = item.get("output", "")
                text = self._format_alpaca(inst, inp, out)
            else:
                text = item.get("text", "")

            # Validate length
            if len(text) < 50:
                return None

            return text

        except StopIteration:
            # Reset stream for next iteration
            self._reset_stream(source)
            return None
        except Exception as e:
            print(f"Error getting sample from {source}: {e}")
            return None

    def _select_source(self) -> str:
        """Select a source based on mixing ratios."""
        r = random.random()
        cumulative = 0.0
        for source, ratio in self.ratios.items():
            cumulative += ratio
            if r < cumulative:
                return source
        return self.sources[-1]

    def __iter__(self):
        """
        Yield batches of tokenized sequences.
        Uses PACKING: accumulates complete texts with EOS until max_length.
        """
        batch = []
        token_buffer = []

        while True:
            # Select source based on ratios
            source = self._select_source()

            # Get text from selected source
            text = self._get_next_sample(source)
            if not text:
                continue

            # Tokenize
            try:
                tokens = self.tokenizer.encode(text)

                # Skip very short sequences
                if len(tokens) < 10:
                    continue

                # Add EOS token after each text
                eos_id = self.tokenizer.eos_token_id
                if eos_id is not None:
                    tokens.append(eos_id)

                # PACKING: Add tokens to buffer
                token_buffer.extend(tokens)

                # When buffer is full, extract training samples
                while len(token_buffer) >= self.max_length:
                    sample_tokens = token_buffer[:self.max_length]
                    token_buffer = token_buffer[self.max_length:]

                    batch.append(torch.tensor(sample_tokens, dtype=torch.long))

                    # Yield batch when full
                    if len(batch) >= self.batch_size:
                        yield self._collate(batch)
                        batch = []

            except Exception:
                continue

    def _collate(self, batch: List[torch.Tensor]) -> torch.Tensor:
        """Collate batch with padding."""
        max_len = max(len(x) for x in batch)

        # Get pad token
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            pad_id = 0

        # Create padded tensor
        padded_batch = torch.full((len(batch), max_len), pad_id, dtype=torch.long)

        for i, x in enumerate(batch):
            padded_batch[i, :len(x)] = x

        return padded_batch


# Convenience classes for each phase
class EllediDatasetPhase1(EllediDatasetV2):
    """Phase 1: English Foundation dataset."""
    def __init__(self, tokenizer, max_length=512, batch_size=4, **kwargs):
        super().__init__(tokenizer, phase=1, max_length=max_length,
                         batch_size=batch_size, **kwargs)


class EllediDatasetPhase2(EllediDatasetV2):
    """Phase 2: Italian Knowledge dataset."""
    def __init__(self, tokenizer, max_length=1024, batch_size=4, **kwargs):
        super().__init__(tokenizer, phase=2, max_length=max_length,
                         batch_size=batch_size, **kwargs)


class EllediDatasetPhase3(EllediDatasetV2):
    """Phase 3: Instruction Alignment dataset."""
    def __init__(self, tokenizer, max_length=1024, batch_size=4, **kwargs):
        super().__init__(tokenizer, phase=3, max_length=max_length,
                         batch_size=batch_size, **kwargs)


if __name__ == "__main__":
    # Self-test
    print("Elleci v2 Dataset Self-Test")
    print("=" * 60)

    # Mock tokenizer for testing
    class MockTokenizer:
        eos_token_id = 0
        pad_token_id = 0
        def encode(self, text):
            return list(range(min(100, len(text) // 4)))

    tokenizer = MockTokenizer()

    # Test each phase
    for phase in [1, 2, 3]:
        print(f"\nTesting Phase {phase}...")
        try:
            ds = EllediDatasetV2(tokenizer, phase=phase, max_length=64, batch_size=2)
            print(f"  Sources: {ds.sources}")
            print(f"  Ratios: {ds.ratios}")
            print(f"  Phase {phase} OK!")
        except Exception as e:
            print(f"  Phase {phase} failed: {e}")

    print("\n" + "=" * 60)
    print("Dataset v2 module ready!")
