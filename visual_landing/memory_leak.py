import torch
import gc

def debug_gpu():
    # Debug out of memory bugs.
    tensor_list = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                tensor_list.append(obj)
                # print(type(obj), obj.size())
        except:
            pass
    print(f'Count of tensors = {len(tensor_list)}.')  