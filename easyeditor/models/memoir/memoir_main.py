from typing import Any, Dict, List, Tuple
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from .MEMOIR import MEMOIR
from .utils import tokenize, get_context_templates
from .memoir_hparams import MEMOIRHyperParams

MEMOIRload = True
def apply_memoir_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: MEMOIRHyperParams,
        copy=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    if copy:
        model = deepcopy(model)
    device = f'cuda:{hparams.device}'
    context_templates = get_context_templates(model, tok, length_params=[[5,5], [10,5]], device=device)
    editor = MEMOIR(model=model, config=hparams, device=device)
    import os
    global MEMOIRload
    if hasattr(hparams, 'load_path') and hparams.load_path and os.path.exists(hparams.load_path) and MEMOIRload:
        print("Start loading the MEMOIR model!")
        editor.load(hparams.load_path)
        MEMOIRload=False
    print(f"Executing MEMOIR algorithm for the update: ")
    for request in requests:
        print(
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
    tokens, act_mask, deact_mask = tokenize(requests, tokenizer=tok, device=device, context_templates=context_templates, hparams=hparams)
    editor.edit(config=hparams, tokens=tokens, act_mask=act_mask, deact_mask=deact_mask)

    weights_copy = editor.reset_layer

    return editor, weights_copy
