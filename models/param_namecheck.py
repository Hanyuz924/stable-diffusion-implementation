def load_diffusion(diffusion_model, official_state_dict):
    my_state_dict = diffusion_model.state_dict()
    aligned_dict = {}
    loaded_count = 0
    mismatch_count = 0
    for hf_key, hf_tensor in official_state_dict.items():
        if "time_embedding" not in hf_key:
            my_key = f"unet.{hf_key}"
        else:
            my_key = hf_key
        if my_key in my_state_dict:
            my_tensor = my_state_dict[my_key]
            if hf_tensor.shape == my_tensor.shape:
                aligned_dict[my_key] = hf_tensor
                loaded_count += 1
            else:
                mismatch_count += 1
                print(f"   mimatch : {my_key}")
                print(f"   hf shape: {list(hf_tensor.shape)}")
                print(f"   my model shape: {list(my_tensor.shape)}\n")
        else:
            print(f"hf key {hf_key} does not exsist in my model: {my_key}")

    print("-" * 70)
    
    load_info = diffusion_model.load_state_dict(aligned_dict, strict=False)
    
    print(f"load count :{loaded_count}.")
    if mismatch_count > 0:
        print(f"number of mismatch count : {mismatch_count}")
    return load_info

def load_decoder(decoder_model, official_state_dict):
    decoder_state_dict = decoder_model.state_dict()
    aligned_dict = {}
    loaded_count = 0
    mismatch_count = 0
    for hf_key, hf_tensor in official_state_dict.items():
        if "decoder" in hf_key:
            my_key = hf_key.replace("decoder.", "")
            if my_key in decoder_state_dict:
                my_tensor = decoder_state_dict[my_key]
                if hf_tensor.shape == my_tensor.shape:
                    aligned_dict[my_key] = hf_tensor
                    loaded_count += 1
                else:
                    mismatch_count += 1
                    print(f"mimatch : {my_key}")
                    print(f"hf shape: {list(hf_tensor.shape)}")
                    print(f"my model shape: {list(my_tensor.shape)}\n")
            else:
                print(f"hf key {hf_key} does not exsist in my model: {my_key}")
    load_info = decoder_model.load_state_dict(aligned_dict, strict=True)
    print(f"load count :{loaded_count}.")
    if mismatch_count > 0:
        print(f"number of mismatch count : {mismatch_count}")
    return load_info

def load_encoder(encoder_model, official_state_dict):
    encoder_state_dict = encoder_model.state_dict()
    aligned_dict = {}
    loaded_count = 0
    mismatch_count = 0
    for hf_key, hf_tensor in official_state_dict.items():
        if "encoder" in hf_key:
            my_key = hf_key.replace("encoder.", "")
            if my_key in encoder_state_dict:
                my_tensor = encoder_state_dict[my_key]
                if hf_tensor.shape == my_tensor.shape:
                    aligned_dict[my_key] = hf_tensor
                    loaded_count += 1
                else:
                    mismatch_count += 1
                    print(f"mimatch : {my_key}")
                    print(f"   hf shape: {list(hf_tensor.shape)}")
                    print(f"   my model shape: {list(my_tensor.shape)}\n")
            else:
                print(f"hf key {hf_key} does not exsist in my model: {my_key}")
    load_info = encoder_model.load_state_dict(aligned_dict, strict=False)
    print(f"load count :{loaded_count}.")
    if mismatch_count > 0:
        print(f"number of mismatch count : {mismatch_count}")
    return load_info

def load_post_quant_conv(post_quant_conv, offical_state_dict):
    post_quant_conv.load_state_dict(offical_state_dict)





