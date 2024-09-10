# This script is from https://gist.github.com/erichans/af745a381b28b1c019f96997ddac4cd7
# based on https://github.com/allenai/longformer/blob/master/scripts/convert_bart_to_longformerencoderdecoder.py


import os
import re

from transformers import BartForConditionalGeneration
from transformers import PLBartTokenizer
from tqdm import tqdm


def create_long_model(
    save_model_to,
    base_model,
    tokenizer_name_or_path,
    max_pos
):
    bart = BartForConditionalGeneration.from_pretrained(base_model)
    model_type = 'allenai/led-base-16384' #if 'large' in base_model else 'allenai/led-base-16384'
    config = LEDConfig.from_pretrained(model_type) #instantiate original LED and change weights and biases
    config.max_encoder_position_embeddings = max_pos
    config.vocab_size = bart.config.vocab_size
    config.attention_window=512
    config.architectures = ["LEDForConditionalGeneration"]
    led = LEDForConditionalGeneration(config=config) #instantiate original LED and change weights and biases

    tokenizer = PLBartTokenizer.from_pretrained(tokenizer_name_or_path, model_max_length=max_pos)

    # in BART attention_probs_dropout_prob is attention_dropout, but LongformerSelfAttention
    # expects attention_probs_dropout_prob, so set it here
    led.config.attention_probs_dropout_prob = bart.config.attention_dropout

    # extend position embeddings
    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos

    copy_weights(bart, led)

    led.save_pretrained(save_model_to)
    tokenizer.save_pretrained(save_model_to)

def copy_weights(bart, led):
    attentions_mapping = {'q_proj': 'query', 'k_proj': 'key', 'v_proj': 'value', 'out_proj': 'output'}

    led_parameters = dict(led.named_parameters())
    bart_parameters = dict(bart.named_parameters())

    parameters_missing_update = list(led_parameters.keys())
    print('Total parameters to copy:', parameters_missing_update)

    for bart_parameter in tqdm(bart_parameters):
        led_parameter = re.sub('^model.', 'led.', bart_parameter) #changing model preffix

        # copy position embeddings over and over to initialize the new position embeddings
        if re.search('model\..*.embed_positions.weight', bart_parameter):
            k = 0
            step = bart.model.encoder.embed_positions.weight.size(0) - 2
            while k < led_parameters[led_parameter].size(0) - 1:
                led_parameters[led_parameter].data[k:(k+step)].copy_(bart_parameters[bart_parameter].data[2:])
                k += step

            parameters_missing_update.remove(led_parameter)
            continue

        # replace the `modeling_bart.SelfAttention` object with `LEDSelfAttention`
        if re.search('model\.encoder.*self_attn\..*_proj', bart_parameter):
            fields = led_parameter.split('.')

            if 'out_proj' in bart_parameter:
                led_parameter = re.sub(fields[-2], attentions_mapping[fields[-2]], led_parameter)
                led_parameters[led_parameter].data.copy_(bart_parameters[bart_parameter].data)
                
                parameters_missing_update.remove(led_parameter)
            else:
                led_parameter_local = re.sub(fields[-2], f'longformer_self_attn.{attentions_mapping[fields[-2]]}', led_parameter)
                led_parameter_global = re.sub(fields[-2], f'longformer_self_attn.{attentions_mapping[fields[-2]]}_global', led_parameter)
                
                led_parameters[led_parameter_local].data.copy_(bart_parameters[bart_parameter].data)
                led_parameters[led_parameter_global].data.copy_(bart_parameters[bart_parameter].data)

                parameters_missing_update.remove(led_parameter_local)
                parameters_missing_update.remove(led_parameter_global)

            continue

        try:
            led_parameters[led_parameter].data.copy_(bart_parameters[bart_parameter].data)
        except:
            print(f"Weights failed to copy {led_parameter}")
        parameters_missing_update.remove(led_parameter)
    
    print('parameters missing update:', parameters_missing_update)
    assert len(parameters_missing_update) == 0

def main():
    save_model_to = f'plbart-led'
    if not os.path.exists(save_model_to):
        os.mkdir(save_model_to)

    create_long_model(
        save_model_to=save_model_to,
        base_model=f'uclanlp/plbart-base',
        tokenizer_name_or_path=f'uclanlp/plbart-base',
        max_pos=2048
    )

if __name__ == '__main__':
    main()
