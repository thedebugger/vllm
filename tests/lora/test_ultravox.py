
from typing import List

import pytest

import vllm
from transformers import  AutoTokenizer
from vllm.assets.audio import AudioAsset
from vllm.lora.request import LoRARequest
from vllm.platforms import current_platform

MODEL_NAME = "fixie-ai/ultravox-v0_3"

VLLM_PLACEHOLDER = "<|reserved_special_token_0|>"

EXPECTED_OUTPUT = [
]

@pytest.fixture(scope="session")
def audio_assets():
    from vllm.assets.audio import AudioAsset
    return [AudioAsset("mary_had_lamb"), AudioAsset("winning_call")]


@pytest.fixture(scope="module", params=("mary_had_lamb", "winning_call"))
def audio(request):
    from vllm.assets.audio import AudioAsset
    return AudioAsset(request.param)

def _get_prompt(audio_count, question, placeholder):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    placeholder = f"{placeholder}\n" * audio_count

    return tokenizer.apply_chat_template([{
        'role': 'user',
        'content': f"{placeholder}{question}"
    }],
                                         tokenize=False,
                                         add_generation_prompt=True)

def do_sample(llm: vllm.LLM, lora_path: str, lora_id: int) -> List[str]:
    sampling_params = vllm.SamplingParams(
        temperature=0,
        max_tokens=5,
    )
    
    inputs = [_get_prompt(1, "Describe the audio above.", VLLM_PLACEHOLDER)]

    outputs = llm.generate(
        inputs,
        sampling_params,
        lora_request=LoRARequest(str(lora_id), lora_id, lora_path)
        if lora_id else None,
    )
    return None


def test_fixie_lora(sql_lora_files):
    llm = vllm.LLM(
        MODEL_NAME,
        max_num_seqs=2,
        enable_lora=True,
        max_loras=4,
        max_lora_rank=64,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        enforce_eager=True
    )
    output1 = do_sample(llm, sql_lora_files, lora_id=1)
    for i in range(len(EXPECTED_OUTPUT)):
        assert EXPECTED_OUTPUT[i].startswith(output1[i])
    return None