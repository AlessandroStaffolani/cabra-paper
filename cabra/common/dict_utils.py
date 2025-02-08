from collections import OrderedDict


def get_config_sub_value(source, key):
    sub_keys = key.split('.')
    tmp = source
    for sub_k in sub_keys:
        if sub_k in tmp:
            tmp = tmp[sub_k]
        else:
            raise KeyError(f'Key sequence "{key}" does not exists in given source')
    return tmp


def set_config_sub_value(source, key, value):
    sub_keys = key.split('.')
    tmp = source
    for i, sub_k in enumerate(sub_keys):
        if sub_k in tmp:
            if i < len(sub_keys) - 1:
                tmp = tmp[sub_k]
            else:
                tmp[sub_k] = value
        else:
            raise KeyError(f'Key sequence "{key}" does not exists in given source')


def sort_dict_keys(source: dict, reverse: bool = False) -> dict:
    return OrderedDict(sorted(source.items(), key=lambda x: x[0], reverse=reverse))
    # sorted_dict = {}
    # for key in sorted(source.keys(), reverse=reverse):
    #     sorted_dict[key] = source[key]
    # return sorted_dict
