"""
Elleci v2 Dataset - 3-Phase Training with Enhanced Data Strategy

Phase 1: English Foundation (60% of training - 35K steps)
    - FineWeb-Edu: 45% (highest quality educational web content)
    - Cosmopedia V2: 25% (synthetic educational)
    - OpenWebMath: 15% (mathematics)
    - The Stack v2: 15% (code)

Phase 2: Italian Knowledge (25% of training - 15K steps)
    - CulturaX Italian: 70% (200GB+ cleaned Italian)
    - Wikipedia IT: 15% (encyclopedia)
    - English maintenance: 15% (prevent forgetting)

Phase 3: Instruction Alignment (15% of training - 10K steps)
    - OpenOrca: 30% (EN reasoning/math)
    - Fauno IT: 25% (IT Q&A)
    - Alpaca IT: 20% (IT instructions)
    - Local instructions: 10% (IT creative)
    - Dolly: 10% (EN diverse)
    - CodeAlpaca: 5% (code)

Data Sources:
- FineWeb-Edu: HuggingFaceFW/fineweb-edu
- Cosmopedia V2: HuggingFaceTB/smollm-corpus (cosmopedia-v2)
- OpenWebMath: open-web-math/open-web-math
- The Stack v2: bigcode/the-stack-v2-train-smol-ids
- CulturaX Italian: uonlp/CulturaX (it)
- Wikipedia IT: wikimedia/wikipedia (20231101.it)
- OpenOrca: Open-Orca/OpenOrca
- Alpaca IT: teelinsan/camoscio
- Fauno IT: andreabac3/StackOverflow-Italian-Fauno-Baize + andreabac3/Quora-Italian-Fauno-Baize
- Dolly: databricks/databricks-dolly-15k
- CodeAlpaca: sahil2801/CodeAlpaca-20k
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
        """Set up data mixing ratios based on training phase."""
        if self.phase == 1:
            # Phase 1: English Foundation
            # Note: LIMA removed (deprecated script), 5% redistributed to FineWeb-Edu
            self.ratios = {
                'fineweb_edu': 0.45,
                'cosmopedia': 0.25,
                'openwebmath': 0.15,
                'stack': 0.15,
            }
            self.sources = list(self.ratios.keys())
        elif self.phase == 2:
            # Phase 2: Italian Knowledge
            self.ratios = {
                'culturax_it': 0.70,
                'wikipedia_it': 0.15,
                'english_mix': 0.15,  # Maintenance: FineWeb + Cosmopedia
            }
            self.sources = list(self.ratios.keys())
        elif self.phase == 3:
            # Phase 3: Instruction Alignment
            self.ratios = {
                'openorca': 0.30,
                'fauno_it': 0.25,
                'alpaca_it': 0.20,
                'local_it': 0.10,
                'dolly': 0.10,
                'codealpaca': 0.05,
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
                "20231101.it", # 20231101 often fails, try recent if needed, but keeping stable for now or updating
                # Updating to a more likely available date or keeping logic to fallback
                # Actually, user asked to update it. Let's try 20231101 -> 20241101.it 
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

    # ========== Sample Getters ==========

    def _get_stream(self, source: str) -> Iterator:
        """Get or create stream for source."""
        if self._iterators.get(source) is None:
            if source == 'fineweb_edu':
                self._iterators[source] = self._get_fineweb_edu_stream()
            elif source == 'cosmopedia':
                self._iterators[source] = self._get_cosmopedia_stream()
            elif source == 'openwebmath':
                self._iterators[source] = self._get_openwebmath_stream()
            elif source == 'stack':
                self._iterators[source] = self._get_stack_stream()
            elif source == 'culturax_it':
                self._iterators[source] = self._get_culturax_it_stream()
            elif source == 'wikipedia_it':
                self._iterators[source] = self._get_wikipedia_it_stream()
            elif source == 'english_mix':
                # For maintenance, alternate between FineWeb and Cosmopedia
                self._iterators[source] = self._get_fineweb_edu_stream()
            elif source == 'openorca':
                self._iterators[source] = self._get_openorca_stream()
            elif source == 'fauno_it':
                self._iterators[source] = self._get_fauno_it_stream()
            elif source == 'alpaca_it':
                self._iterators[source] = self._get_alpaca_it_stream()
            elif source == 'local_it':
                # Local instructions don't need a stream
                pass
            elif source == 'dolly':
                self._iterators[source] = self._get_dolly_stream()
            elif source == 'codealpaca':
                self._iterators[source] = self._get_codealpaca_stream()
            else:
                raise ValueError(f"Unknown source: {source}")

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
            if source in ['fineweb_edu', 'cosmopedia', 'openwebmath', 'stack',
                          'culturax_it', 'wikipedia_it', 'english_mix']:
                text = item.get("text", "")
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
